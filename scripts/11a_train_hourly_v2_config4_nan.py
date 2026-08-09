"""
Adım 11a: KONFIGÜRASYON 4 eğitimi — sayısal kararsızlık / NaN başarısızlık modu.

>>> BU SCRIPT BİR REKONSTRÜKSİYONDUR <<<
Makalenin Bölüm 4.4'te raporladığı Config 4 başarısızlık modunu üretir.
Config 5 eğitim script'inden (11_train_hourly_v2.py) türetilmiştir; Config 5'te
eklenen KARARLILIK DÜZELTMELERİ burada BİLEREK GERİ ALINMIŞTIR, çünkü Config 4
bu düzeltmelerden ÖNCEki durumdur (makale Bölüm 3.6):

  Config 5 düzeltmesi                         | Config 4 (bu dosya)
  ------------------------------------------- | --------------------------------
  Kenar ağırlığı: log1p + min-max (pozitif)   | z-score (negatif üretir)  <-- asıl neden
  Her konvolüsyon sonrası LayerNorm           | LayerNorm YOK
  Gradyan norm kırpma (max_norm=1.0)          | Kırpma YOK
  GAT learning rate = 0.001                   | GAT learning rate = 0.005

Beklenen sonuç (makale Tablo, Bölüm 4.4):  MLP calisir, GCN = NaN, GAT ~ kararsiz.
Makaledeki raporlanan degerler:  MLP 8.957 | GCN NaN | GAT 9.988

Girdi: pyg_dataset_hourly_v2_config4_zscore.pt  (10a_...config4.py çıktısı)

NOT: Bu script model AĞIRLIĞI KAYDETMEZ; amacı reprodüksiyon için başarısızlık
modunu göstermektir (GCN için kaydedilecek geçerli bir ağırlık yoktur).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader

torch.manual_seed(42)

obj = torch.load("pyg_dataset_hourly_v2_config4_zscore.pt", weights_only=False)
train_data = obj["train"]
test_data = obj["test"]
target_mean = obj["target_mean"]
target_std = obj["target_std"]
in_channels = train_data[0].x.shape[1]
print(f"Giriş boyutu (WINDOW * n_features): {in_channels}")

# batch size 8 — makale Bölüm 3.9 (tüm beş konfigürasyon için 8)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)


# ---- Config 4 modelleri: LayerNorm YOK (Config 5'te eklenen düzeltme geri alındı) ----
class GCNDelayPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_weight=None):
        # NOT: edge_weight z-score olduğu için negatif değerler içerir; GCNConv'un
        # simetrik derece normalizasyonu (negatif toplamın karekökü) NaN üretir.
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_weight)
        h = F.relu(h)
        return self.out(h).squeeze(-1)


class GATDelayPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, heads=4, edge_dim=1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.2, edge_dim=edge_dim)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.conv1(x, edge_index, edge_attr)
        h = F.elu(h)
        h = self.conv2(h, edge_index, edge_attr)
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


def evaluate(model, loader, use_edge=True, gat=False):
    model.eval()
    total_err = 0.0
    total_mask = 0.0
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


def train_model(model, name, epochs=80, lr=0.005, use_edge=True, gat=False):
    # Config 4: gradyan kırpma YOK (Config 5'te eklenen düzeltme geri alındı)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_mae = float("inf")
    nan_hit = False

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
            # NaN tespiti (Config 4'ün beklenen başarısızlık modu)
            if not torch.isfinite(loss):
                nan_hit = True
                break
            loss.backward()
            optimizer.step()
        if nan_hit:
            print(f"[{name}] Epoch {epoch}: kayıp NaN/Inf -> eğitim başarısız "
                  f"(makale Bölüm 4.4'teki beklenen davranış).")
            return float("nan")
        if epoch % 10 == 0 or epoch == 1:
            test_mae = evaluate(model, test_loader, use_edge, gat)
            if np.isfinite(test_mae):
                best_mae = min(best_mae, test_mae)
            print(f"[{name}] Epoch {epoch:3d} | Test MAE (dk): {test_mae:.3f}")

    print(f"[{name}] En iyi Test MAE (dakika): {best_mae:.3f}\n")
    return best_mae


results = {}

print("=" * 55)
print("1) MLP (graf yapısı YOK)")
print("=" * 55)
results["MLP"] = train_model(MLPBaseline(in_channels), "MLP", use_edge=False)

print("=" * 55)
print("2) GCN (z-score kenar ağırlıkları -> NaN bekleniyor)")
print("=" * 55)
results["GCN"] = train_model(GCNDelayPredictor(in_channels), "GCN", use_edge=True, gat=False)

print("=" * 55)
print("3) GAT (z-score kenar ağırlıkları -> kararsız)")
print("=" * 55)
results["GAT"] = train_model(GATDelayPredictor(in_channels), "GAT", use_edge=True, gat=True, lr=0.005)

print("=" * 55)
print("KONFIGÜRASYON 4 ÖZET (makale Bölüm 4.4)")
print("=" * 55)
for k, v in results.items():
    print(f"  {k:5s}: {v}")
print("Beklenen: MLP sonlu, GCN NaN, GAT kararsız/yüksek.")
