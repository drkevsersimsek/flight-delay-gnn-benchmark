"""Takvim saati izgarasi (Section 3.5, Table 1 row 5c): graphs_hourly.pkl + graphs_hourly_v2.pkl -> graphs_hourly_cal.npz"""
import pickle, numpy as np, pandas as pd
G = pickle.load(open("graphs_hourly.pkl", "rb"))["graphs"]
v2 = pickle.load(open("graphs_hourly_v2.pkl", "rb")); EI = v2["fixed_edge_index"]; N = v2["n_nodes"]
pos = {(int(a), int(b)): k for k, (a, b) in enumerate(EI.T)}
ts = pd.to_datetime([g["timeslot"] for g in G])
grid = pd.date_range("2019-01-01 00:00", ts.max(), freq="h"); T = len(grid); ti = {t: k for k, t in enumerate(grid)}
X = np.zeros((T, N, 4), np.float32); EW = np.zeros((T, EI.shape[1]), np.float32)
for g, t in zip(G, ts):
    k = ti[t]; X[k] = g["node_feat"]
    for (a, b), w in zip(g["edge_index"].T, g["edge_weight"]):
        p = pos.get((int(a), int(b)))
        if p is not None: EW[k, p] = w
has = X[:, :, 0] > 0
target = np.where(has[1:], X[1:, :, 1], np.nan).astype(np.float32); mask = has[1:].astype(np.float32)
X, EW, has, grid = X[:-1], EW[:-1], has[:-1], grid[:-1]
np.savez_compressed("graphs_hourly_cal.npz", node_feat=X, edge_weight=EW, target=target, target_mask=mask, has_dep=has,
    timeslot=grid.values.astype("datetime64[ns]").astype("int64"), fixed_edge_index=EI, n_nodes=np.array(N))
print(len(grid), "snapshot; maskeli hedef orani %.1f%%" % (100 * (1 - mask.mean())))
