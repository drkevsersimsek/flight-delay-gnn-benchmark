"""
Step 10b: graphs_hourly_v2.pkl -> PyG Data objects (LEAKAGE-FREE variant).

This is identical to 10_prepare_pyg_hourly_v2.py EXCEPT that every
standardisation statistic (node-feature mean/std, edge-weight log1p
min/max, and target mean/std) is fit ONLY on the training-visible
graphs, i.e. graphs strictly before the held-out test period. No
test-period snapshot contributes to any normalisation statistic.

This removes the mild normalisation leakage described in Section 3.5 /
5.4 of the paper. Output: pyg_dataset_hourly_v2_trainonly.pt, which is a
drop-in replacement for pyg_dataset_hourly_v2.pt in 12_multi_seed_v2.py
(see 12b_multi_seed_v2_trainonly.py).
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

# The chronological 80/20 split is taken over the windowed samples.
# Samples are indexed i = WINDOW-1 .. len(graphs)-1; split_idx marks the
# first test sample. The last graph index touched by any TRAINING sample
# (its window or its target) is (split_idx + WINDOW - 2); therefore
# graphs[:split_idx + WINDOW - 1] are exactly the training-visible graphs
# and graphs[split_idx + WINDOW - 1:] are test-only.
n_samples = len(graphs) - (WINDOW - 1)
split_idx = int(n_samples * 0.8)
train_max_graph = split_idx + WINDOW - 1  # exclusive end of training-visible graphs

# ---- statistics fit on TRAINING-VISIBLE graphs only ----
train_graphs = graphs[:train_max_graph]
all_feats = np.concatenate([g["node_feat"] for g in train_graphs], axis=0)
feat_mean = all_feats.mean(axis=0)
feat_std = all_feats.std(axis=0) + 1e-6

all_edge_w = np.concatenate([g["edge_weight"] for g in train_graphs], axis=0)
all_edge_w_log = np.log1p(all_edge_w)
edge_w_min = all_edge_w_log.min()
edge_w_max = all_edge_w_log.max()
edge_w_range = (edge_w_max - edge_w_min) + 1e-6

# target stats over TRAINING SAMPLE targets only (graphs WINDOW-1 .. train_max_graph-1)
target_graphs = graphs[WINDOW - 1:train_max_graph]
all_targets = np.concatenate([g["target"][~np.isnan(g["target"])] for g in target_graphs])
target_mean = float(all_targets.mean())
target_std = float(all_targets.std() + 1e-6)

print("Feature mean:", feat_mean)
print("Feature std:", feat_std)
print("Target mean / std (train-only):", target_mean, target_std)
print("Edge weight (log1p+minmax) range: [0.01, 1.01]")

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
    edge_w = edge_w + 0.01  # small floor for all-zero edges (numerical stability)

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

split_idx_check = int(len(pyg_data_list) * 0.8)
train_data = pyg_data_list[:split_idx_check]
test_data = pyg_data_list[split_idx_check:]

print(f"Window: {WINDOW} hours")
print(f"Train graphs: {len(train_data)}")
print(f"Test graphs:  {len(test_data)}")
print(f"Input feature dim (x): {pyg_data_list[0].x.shape}")
print(f"Fixed edge count: {edge_index_t.shape[1]}")

torch.save({
    "train": train_data,
    "test": test_data,
    "n_nodes": n_nodes,
    "window": WINDOW,
    "n_features_per_step": 4,
    "target_mean": target_mean,
    "target_std": target_std,
}, "pyg_dataset_hourly_v2_trainonly.pt")

print("Saved pyg_dataset_hourly_v2_trainonly.pt")
