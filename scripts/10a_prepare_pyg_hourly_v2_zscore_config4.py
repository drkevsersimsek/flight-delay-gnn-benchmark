"""
Adım 10a: KONFIGÜRASYON 4 (sayısal kararsızlık / NaN başarısızlık modu).

>>> BU SCRIPT BİR REKONSTRÜKSİYONDUR <<<
Makalenin ilk gönderimindeki Config 4 hazırlık script'i orijinal yükleme
dosyaları arasında yer almıyordu (yalnızca 09_build_graphs_hourly_v2.py
içindeki bir yorumda referans veriliyordu). Bu dosya, makalede Bölüm 3.6
ve 4.4'te tarif edilen yöntemi birebir uygulamak üzere Config 5 hazırlık
script'inden (10_prepare_pyg_hourly_v2.py) türetilmiştir: TEK fark, kenar
ağırlığı normalizasyonudur.

Config 5 (kararlı):   log1p + min-max -> [0.01, 1.01]  (kesinlikle pozitif)
Config 4 (bu dosya):  z-score standardizasyon (sıfır ortalama, birim varyans)

Uçuş sayıları negatif olmadığından ve sağa çarpık olduğundan, z-score
standardizasyonu kenar ağırlıklarının önemli bir kısmını NEGATİF yapar.
Bu negatif kenar ağırlıkları GCNConv'un simetrik derece normalizasyonuna
(negatif toplamın karekökü) girdiğinde ileri geçişte NaN üretir
(bkz. 11a_train_hourly_v2_config4_nan.py ve makale Bölüm 4.4).

Girdi : graphs_hourly_v2.pkl  (09_build_graphs_hourly_v2.py çıktısı — Config 5 ile ORTAK)
Çıktı : pyg_dataset_hourly_v2_config4_zscore.pt
"""

import pickle
import numpy as np
import torch
from torch_geometric.data import Data

WINDOW = 24

with open("graphs_hourly_v2.pkl", "rb") as f:
    obj = pickle.load(f)

graphs = obj["graphs"]
n_nodes = obj["n_nodes"]
fixed_edge_index = obj["fixed_edge_index"]

# Düğüm öznitelikleri: Config 5 ile aynı (tüm-veri z-score) — burada tartışılan
# kararsızlık düğüm özniteliklerinden değil, KENAR ağırlıklarından kaynaklanır.
all_feats = np.concatenate([g["node_feat"] for g in graphs], axis=0)
feat_mean = all_feats.mean(axis=0)
feat_std = all_feats.std(axis=0) + 1e-6

# ---- CONFIG 4 FARKI: kenar ağırlıklarına z-score standardizasyon ----
# (Config 5'te bunun yerine log1p + min-max kullanılır.)
all_edge_w = np.concatenate([g["edge_weight"] for g in graphs], axis=0)
edge_w_mean = all_edge_w.mean()
edge_w_std = all_edge_w.std() + 1e-6

frac_negative = float(((all_edge_w - edge_w_mean) / edge_w_std < 0).mean())
print("=" * 60)
print("KONFIGÜRASYON 4  (z-score kenar normalizasyonu — NaN failure mode)")
print("=" * 60)
print(f"Kenar ağırlığı ortalama / std : {edge_w_mean:.4f} / {edge_w_std:.4f}")
print(f"z-score sonrası NEGATİF kenar ağırlığı oranı: {frac_negative:.1%}")
print("UYARI: Negatif kenar ağırlıkları GCNConv'da NaN üretecektir "
      "(makale Bölüm 4.4). Bu, bu konfigürasyonun beklenen (başarısız) davranışıdır.")

all_targets = np.concatenate([g["target"][~np.isnan(g["target"])] for g in graphs])
target_mean = all_targets.mean()
target_std = all_targets.std() + 1e-6

edge_index_t = torch.tensor(fixed_edge_index, dtype=torch.long)

pyg_data_list = []
for i in range(WINDOW - 1, len(graphs)):
    window_feats = []
    for j in range(i - WINDOW + 1, i + 1):
        x = (graphs[j]["node_feat"] - feat_mean) / feat_std
        window_feats.append(x)
    x_concat = np.concatenate(window_feats, axis=1)

    # z-score kenar ağırlığı (Config 4) — negatif değerler ÜRETİR
    edge_w = (graphs[i]["edge_weight"] - edge_w_mean) / edge_w_std

    target = graphs[i]["target"]
    mask = graphs[i]["target_mask"]
    y = np.where(mask > 0, (target - target_mean) / target_std, 0.0).astype(np.float32)

    data = Data(
        x=torch.tensor(x_concat, dtype=torch.float32),
        edge_index=edge_index_t,
        edge_attr=torch.tensor(edge_w, dtype=torch.float32).unsqueeze(-1),
        y=torch.tensor(y, dtype=torch.float32),
        mask=torch.tensor(mask, dtype=torch.float32),
    )
    pyg_data_list.append(data)

split_idx = int(len(pyg_data_list) * 0.8)
train_data = pyg_data_list[:split_idx]
test_data = pyg_data_list[split_idx:]

print(f"Pencere boyutu: {WINDOW} saat")
print(f"Train graf sayısı: {len(train_data)}")
print(f"Test graf sayısı:  {len(test_data)}")
print(f"Giriş öznitelik boyutu (x): {pyg_data_list[0].x.shape}")
print(f"Sabit kenar sayısı: {edge_index_t.shape[1]}")

torch.save({
    "train": train_data,
    "test": test_data,
    "n_nodes": n_nodes,
    "window": WINDOW,
    "n_features_per_step": 4,
    "target_mean": target_mean,
    "target_std": target_std,
}, "pyg_dataset_hourly_v2_config4_zscore.pt")

print("pyg_dataset_hourly_v2_config4_zscore.pt kaydedildi.")
