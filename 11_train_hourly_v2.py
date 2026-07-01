"""
Adım 11: İyileştirilmiş eğitim - sabit graf topolojisi, edge_weight kullanımı,
maskeli loss (sadece gerçek uçuşu olan düğümler için hata hesaplanır).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader

torch.manual_seed(42)

obj = torch.load("pyg_dataset_hourly_v2.pt", weights_only=False)
train_data = obj["train"]
test_data = obj["test"]
target_mean = obj["target_mean"]
target_std = obj["target_std"]
in_channels = train_data[0].x.shape[1]
print(f"Giriş boyutu (WINDOW * n_features): {in_channels}")

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)


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


def masked_mae_minutes(pred, y, mask):
    pred_min = pred * target_std + target_mean
    true_min = y * target_std + target_mean
    diff = (pred_min - true_min).abs() * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def train_model(model, name, train_loader, test_loader, epochs=80, lr=0.005, use_edge=True, gat=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_mae = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_graphs = 0
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
            total_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs

        if epoch % 10 == 0 or epoch == 1:
            train_loss = total_loss / n_graphs
            test_mae = evaluate(model, test_loader, use_edge, gat)
            best_mae = min(best_mae, test_mae)
            print(f"[{name}] Epoch {epoch:3d} | Train MaskedMSE: {train_loss:.4f} | Test MAE (dk): {test_mae:.3f}")

    print(f"[{name}] En iyi Test MAE (dakika): {best_mae:.3f}\n")
    return best_mae


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


results = {}

print("=" * 55)
print("1) Baseline: MLP (graf yapısı YOK, maskeli loss)")
print("=" * 55)
mlp = MLPBaseline(in_channels)
results["MLP"] = train_model(mlp, "MLP", train_loader, test_loader, use_edge=False)

print("=" * 55)
print("2) GCN (sabit graf + edge_weight, maskeli loss)")
print("=" * 55)
gcn = GCNDelayPredictor(in_channels)
results["GCN"] = train_model(gcn, "GCN", train_loader, test_loader, use_edge=True, gat=False)

print("=" * 55)
print("3) GAT (sabit graf + edge_attr, maskeli loss)")
print("=" * 55)
gat_model = GATDelayPredictor(in_channels)
results["GAT"] = train_model(gat_model, "GAT", train_loader, test_loader, use_edge=True, gat=True, lr=0.001)

print("=" * 55)
print("ÖZET (en iyi Test MAE, dakika - düşük daha iyi)")
print("=" * 55)
for k, v in results.items():
    print(f"  {k:5s}: {v:.3f}")

torch.save(mlp.state_dict(), "mlp_hourly_v2_model.pt")
torch.save(gcn.state_dict(), "gcn_hourly_v2_model.pt")
torch.save(gat_model.state_dict(), "gat_hourly_v2_model.pt")
print("\nModeller kaydedildi: mlp_hourly_v2_model.pt, gcn_hourly_v2_model.pt, gat_hourly_v2_model.pt")
