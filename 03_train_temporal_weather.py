"""
Adım 3: Zamansal pencereli GCN modeli (Spatio-Temporal yaklaşımına basit bir giriş).

Giriş: x.shape = [n_nodes, WINDOW * n_features] -> son WINDOW günün öznitelikleri
       birleştirilmiş halde her düğüme verilir.
Graf yapısı: o günün havalimanı bağlantı grafı (edge_index).

Baseline: aynı girdiyi kullanan ama graf yapısını yok sayan MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader

torch.manual_seed(42)

obj = torch.load("pyg_dataset_temporal_weather.pt", weights_only=False)
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
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        return self.out(h).squeeze(-1)


class GATDelayPredictor(nn.Module):
    """Graph Attention Network - hangi komşu havalimanının daha etkili olduğunu öğrenir."""
    def __init__(self, in_channels, hidden_channels=32, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.2)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
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

    def forward(self, x, edge_index=None):
        return self.net(x).squeeze(-1)


def train_model(model, name, train_loader, test_loader, epochs=80, lr=0.005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = nn.MSELoss()
    best_mae = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        if epoch % 10 == 0 or epoch == 1:
            train_loss = total_loss / len(train_loader.dataset)
            test_mae = evaluate(model, test_loader)
            best_mae = min(best_mae, test_mae)
            print(f"[{name}] Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | Test MAE (dk): {test_mae:.3f}")

    print(f"[{name}] En iyi Test MAE (dakika): {best_mae:.3f}\n")
    return best_mae


def evaluate(model, loader):
    model.eval()
    total_abs_err = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch.x, batch.edge_index)
            pred_min = pred * target_std + target_mean
            true_min = batch.y * target_std + target_mean
            total_abs_err += (pred_min - true_min).abs().sum().item()
            n += pred.numel()
    return total_abs_err / n


results = {}

print("=" * 55)
print("1) Baseline: MLP (zamansal pencere, graf yapısı YOK)")
print("=" * 55)
mlp = MLPBaseline(in_channels)
results["MLP"] = train_model(mlp, "MLP", train_loader, test_loader)

print("=" * 55)
print("2) GCN (zamansal pencere + havalimanı graf yapısı)")
print("=" * 55)
gcn = GCNDelayPredictor(in_channels)
results["GCN"] = train_model(gcn, "GCN", train_loader, test_loader)

print("=" * 55)
print("3) GAT (zamansal pencere + attention tabanlı graf yapısı)")
print("=" * 55)
gat = GATDelayPredictor(in_channels)
results["GAT"] = train_model(gat, "GAT", train_loader, test_loader)

print("=" * 55)
print("ÖZET (en iyi Test MAE, dakika - düşük daha iyi)")
print("=" * 55)
for k, v in results.items():
    print(f"  {k:5s}: {v:.3f}")

torch.save(mlp.state_dict(), "mlp_temporal_weather_model.pt")
torch.save(gcn.state_dict(), "gcn_temporal_weather_model.pt")
torch.save(gat.state_dict(), "gat_temporal_weather_model.pt")
print("\nModeller kaydedildi: mlp_temporal_weather_model.pt, gcn_temporal_weather_model.pt, gat_temporal_weather_model.pt")
