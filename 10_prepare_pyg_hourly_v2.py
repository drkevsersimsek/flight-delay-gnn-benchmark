"""
Adım 10: graphs_hourly_v2.pkl -> PyG Data objeleri.

x.shape = [n_nodes, WINDOW * 4]  (son WINDOW saatin öznitelikleri)
edge_index: sabit (tüm snapshot'larda aynı)
edge_attr (edge_weight): o saate ait, normalize edilmiş uçuş yoğunluğu
y: hedef gecikme (normalize), target_mask: hangi düğümlerin hedefi geçerli
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

all_feats = np.concatenate([g["node_feat"] for g in graphs], axis=0)
feat_mean = all_feats.mean(axis=0)
feat_std = all_feats.std(axis=0) + 1e-6

all_edge_w = np.concatenate([g["edge_weight"] for g in graphs], axis=0)
# log1p ile sıkıştır, sonra [0,1] aralığına min-max normalize et (GCNConv negatif
# kenar ağırlıklarıyla NaN üretebildiği için z-score KULLANILMAZ)
all_edge_w_log = np.log1p(all_edge_w)
edge_w_min = all_edge_w_log.min()
edge_w_max = all_edge_w_log.max()
edge_w_range = (edge_w_max - edge_w_min) + 1e-6

all_targets = np.concatenate([g["target"][~np.isnan(g["target"])] for g in graphs])
target_mean = all_targets.mean()
target_std = all_targets.std() + 1e-6

print("Öznitelik ortalama:", feat_mean)
print("Öznitelik std:", feat_std)
print("Hedef ortalama / std:", target_mean, target_std)
print("Edge weight (log1p+minmax) aralığı: [0.01, 1.01]")

edge_index_t = torch.tensor(fixed_edge_index, dtype=torch.long)

pyg_data_list = []
for i in range(WINDOW - 1, len(graphs)):
    window_feats = []
    for j in range(i - WINDOW + 1, i + 1):
        x = (graphs[j]["node_feat"] - feat_mean) / feat_std
        window_feats.append(x)
    x_concat = np.concatenate(window_feats, axis=1)

    edge_w_log = np.log1p(graphs[i]["edge_weight"])
    edge_w = (edge_w_log - edge_w_min) / edge_w_range  # [0, 1]
    edge_w = edge_w + 0.01  # tamamen 0 olan kenarlar için küçük taban (sayısal kararlılık)

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
}, "pyg_dataset_hourly_v2.pt")

print("pyg_dataset_hourly_v2.pt kaydedildi.")
