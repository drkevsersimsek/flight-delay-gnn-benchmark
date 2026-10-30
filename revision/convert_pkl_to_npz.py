"""graphs_hourly_v2.pkl -> graphs_hourly_v2.npz (NumPy surumunden bagimsiz). NumPy >= 2 gerekir."""
import pickle, numpy as np
o = pickle.load(open("graphs_hourly_v2.pkl", "rb")); G = o["graphs"]
np.savez_compressed("graphs_hourly_v2.npz",
    node_feat=np.stack([g["node_feat"] for g in G]), edge_weight=np.stack([g["edge_weight"] for g in G]),
    target=np.stack([g["target"] for g in G]), target_mask=np.stack([g["target_mask"] for g in G]),
    has_dep=np.stack([g["has_dep"] for g in G]),
    timeslot=np.array([np.datetime64(g["timeslot"], "ns").astype("int64") for g in G]),
    fixed_edge_index=o["fixed_edge_index"], n_nodes=np.array(o["n_nodes"]))
print("graphs_hourly_v2.npz kaydedildi")
