"""
Adım 12: Konfigürasyon 5 (saatlik, sabit graf, edge_weight, maskeli loss) için
5 farklı random seed ile eğitimi tekrarlayıp, her model için
ortalama Test MAE ± standart sapma raporlama.

Bu script 11_train_hourly_v2.py ile AYNI model/eğitim mantığını kullanır,
sadece dış bir döngü ile 5 seed üzerinden tekrarlar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader

SEEDS = [42, 1, 7, 123, 2024]

obj = torch.load("pyg_dataset_hourly_v2.pt", weights_only=False)
train_data = obj["train"]
test_data = obj["test"]
target_mean = obj["target_mean"]
target_std = obj["target_std"]
in_channels = train_data[0].x.shape[1]
print(f"Giriş boyutu (WINDOW * n_features): {in_channels}")
print(f"Seed'ler: {SEEDS}\n")


class GCNDelayPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.out = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_weight)
        h = self.norm2(h)
        h = F.relu(h)
        return self.out(h).squeeze(-1)


class GATDelayPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, heads=4, edge_dim=1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2, edge_dim=edge_dim)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.2, edge_dim=edge_dim)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.conv1(x, edge_index, edge_attr)
        h = self.norm1(h)
        h = F.elu(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = self.norm2(h)
        h = F.elu(h)
        return self.out(h).squeeze(-1)


class MLPBaseline(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index=None, edge_attr=None):
        return self.net(x).squeeze(-1)


def masked_mse(pred, y, mask):
    diff = (pred - y) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def train_model(model, name, train_loader, val_loader, test_loader, epochs=80, lr=0.005, use_edge=True, gat=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val_mae = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            if use_edge:
                if gat:
                    pred = model(batch.x, batch.edge_index, batch.edge_attr)
                else:
                    pred = model(batch.x, batch.edge_index, batch.edge_attr.squeeze(-1))
            else:
                pred = model(batch.x, batch.edge_index)
            loss = masked_mse(pred, batch.y, batch.mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            val_mae = evaluate(model, val_loader, use_edge, gat)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # En iyi (validation'a göre) modeli geri yükle, test'i SADECE bu noktada değerlendir
    if best_state is not None:
        model.load_state_dict(best_state)
    test_mae = evaluate(model, test_loader, use_edge, gat)
    return test_mae, best_val_mae, best_epoch


def evaluate(model, loader, use_edge=True, gat=False):
    model.eval()
    total_err = 0
    total_mask = 0
    with torch.no_grad():
        for batch in loader:
            if use_edge:
                if gat:
                    pred = model(batch.x, batch.edge_index, batch.edge_attr)
                else:
                    pred = model(batch.x, batch.edge_index, batch.edge_attr.squeeze(-1))
            else:
                pred = model(batch.x, batch.edge_index)
            pred_min = pred * target_std + target_mean
            true_min = batch.y * target_std + target_mean
            err = (pred_min - true_min).abs() * batch.mask
            total_err += err.sum().item()
            total_mask += batch.mask.sum().item()
    return total_err / max(total_mask, 1.0)


# Her model için tüm seed sonuçlarını topla
all_results = {"MLP": [], "GCN": [], "GAT": []}
all_val = {"MLP": [], "GCN": [], "GAT": []}
all_best_epoch = {"MLP": [], "GCN": [], "GAT": []}

# Kronolojik train -> train_inner (%80) / validation (%20) bölmesi.
# (test_data zaten orijinal pipeline'da ayrılmış %20'lik dilim; burada
#  sadece train_data'nın son kısmını validation için ayırıyoruz.)
val_split_idx = int(len(train_data) * 0.8)
train_inner_data = train_data[:val_split_idx]
val_data = train_data[val_split_idx:]
print(f"Train (inner): {len(train_inner_data)}  |  Validation: {len(val_data)}  |  Test: {len(test_data)}\n")

for run_idx, seed in enumerate(SEEDS, 1):
    print("=" * 60)
    print(f"RUN {run_idx}/{len(SEEDS)}  (seed = {seed})")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_inner_data, batch_size=8, shuffle=True, generator=g)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

    torch.manual_seed(seed)
    mlp = MLPBaseline(in_channels)
    test_mae, val_mae, best_epoch = train_model(mlp, "MLP", train_loader, val_loader, test_loader, use_edge=False)
    all_results["MLP"].append(test_mae)
    all_val["MLP"].append(val_mae)
    all_best_epoch["MLP"].append(best_epoch)
    print(f"  MLP -> Val MAE: {val_mae:.3f} (epoch {best_epoch}) | Test MAE: {test_mae:.3f}")

    torch.manual_seed(seed)
    gcn = GCNDelayPredictor(in_channels)
    test_mae, val_mae, best_epoch = train_model(gcn, "GCN", train_loader, val_loader, test_loader, use_edge=True, gat=False)
    all_results["GCN"].append(test_mae)
    all_val["GCN"].append(val_mae)
    all_best_epoch["GCN"].append(best_epoch)
    print(f"  GCN -> Val MAE: {val_mae:.3f} (epoch {best_epoch}) | Test MAE: {test_mae:.3f}")

    torch.manual_seed(seed)
    gat_model = GATDelayPredictor(in_channels)
    test_mae, val_mae, best_epoch = train_model(gat_model, "GAT", train_loader, val_loader, test_loader, use_edge=True, gat=True, lr=0.001)
    all_results["GAT"].append(test_mae)
    all_val["GAT"].append(val_mae)
    all_best_epoch["GAT"].append(best_epoch)
    print(f"  GAT -> Val MAE: {val_mae:.3f} (epoch {best_epoch}) | Test MAE: {test_mae:.3f}\n")

# ===================== \u00d6ZET =====================
print("=" * 60)
print(f"\u00d6ZET - {len(SEEDS)} seed \u00fczerinden Test MAE (dakika, d\u00fc\u015f\u00fck daha iyi)")
print("Not: 'best epoch' validation set'e g\u00f6re se\u00e7ildi; Test MAE bu epoch'ta TEK SEFER \u00f6l\u00e7\u00fcld\u00fc.")
print("=" * 60)
print(f"{'Model':<8}{'Mean':>10}{'Std':>10}{'  T\u00fcm de\u011ferler'}")
for model_name, values in all_results.items():
    values = np.array(values)
    mean = values.mean()
    std = values.std(ddof=1)
    vals_str = ", ".join(f"{v:.3f}" for v in values)
    print(f"{model_name:<8}{mean:>10.3f}{std:>10.3f}  [{vals_str}]")

print("\nSe\u00e7ilen 'best epoch' de\u011ferleri (validation'a g\u00f6re):")
for model_name, epochs_list in all_best_epoch.items():
    print(f"  {model_name}: {epochs_list}")

print("\nKopyalanabilir özet (ortalama ± std):")
for model_name, values in all_results.items():
    values = np.array(values)
    print(f"  {model_name}: {values.mean():.3f} ± {values.std(ddof=1):.3f}")
