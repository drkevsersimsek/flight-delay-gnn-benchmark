"""
Adım 2: Graf snapshot'larını zamansal pencere (sliding window) ile PyTorch Geometric
Data objelerine çevirme.

Her örnek = son WINDOW gündeki düğüm özniteliklerinin dizisi + o günün graf yapısı
(en güncel günün edge_index/edge_weight'i kullanılır).
Hedef = bir sonraki günün ortalama kalkış gecikmesi.

x.shape = [n_nodes, WINDOW * n_features]  (geçmiş günler kanal olarak birleştirilir)
"""

import pickle
import numpy as np
import torch
from torch_geometric.data import Data

WINDOW = 7  # son kaç günün öznitelikleri kullanılacak

with open("graphs_weather.pkl", "rb") as f:
    obj = pickle.load(f)

graphs = obj["graphs"]
n_nodes = obj["n_nodes"]
n_features = graphs[0]["node_feat"].shape[1]
print(f"Düğüm başına öznitelik sayısı: {n_features}")

# Normalizasyon istatistikleri
all_feats = np.concatenate([g["node_feat"] for g in graphs], axis=0)
feat_mean = all_feats.mean(axis=0)
feat_std = all_feats.std(axis=0) + 1e-6

all_targets = np.concatenate([g["target"] for g in graphs], axis=0)
target_mean = all_targets.mean()
target_std = all_targets.std() + 1e-6

print("Öznitelik ortalama:", feat_mean)
print("Öznitelik std:", feat_std)
print("Hedef ortalama / std:", target_mean, target_std)

pyg_data_list = []
for i in range(WINDOW - 1, len(graphs)):
    # Son WINDOW günün normalize edilmiş özniteliklerini yan yana koy
    window_feats = []
    for j in range(i - WINDOW + 1, i + 1):
        x = (graphs[j]["node_feat"] - feat_mean) / feat_std
        window_feats.append(x)
    x_concat = np.concatenate(window_feats, axis=1)  # [n_nodes, WINDOW * 4]

    # En güncel günün graf yapısını kullan
    edge_index = graphs[i]["edge_index"]
    edge_weight = graphs[i]["edge_weight"]

    y = (graphs[i]["target"] - target_mean) / target_std

    data = Data(
        x=torch.tensor(x_concat, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_weight, dtype=torch.float32).unsqueeze(-1),
        y=torch.tensor(y, dtype=torch.float32),
    )
    pyg_data_list.append(data)

split_idx = int(len(pyg_data_list) * 0.8)
train_data = pyg_data_list[:split_idx]
test_data = pyg_data_list[split_idx:]

print(f"Pencere boyutu: {WINDOW} gün")
print(f"Train graf sayısı: {len(train_data)}")
print(f"Test graf sayısı:  {len(test_data)}")
print(f"Giriş öznitelik boyutu (x): {pyg_data_list[0].x.shape}")

torch.save({
    "train": train_data,
    "test": test_data,
    "n_nodes": n_nodes,
    "window": WINDOW,
    "n_features_per_step": n_features,
    "target_mean": target_mean,
    "target_std": target_std,
}, "pyg_dataset_temporal_weather.pt")

print("pyg_dataset_temporal_weather.pt kaydedildi.")
