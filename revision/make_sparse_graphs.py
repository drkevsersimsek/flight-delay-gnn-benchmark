"""Seyrek/yonlu graflar (Section 4.3, Table 1 row 5s): her havalimani icin egitim donemindeki en yogun k gelen rota."""
import numpy as np
z = np.load("graphs_hourly_v2.npz"); a = {k: z[k] for k in z.files}
T = len(a["timeslot"]); W = 24; n = T - W + 1; last = W - 1 + int(n * .8) - 1
vol = a["edge_weight"][:last + 1].sum(0); EI = a["fixed_edge_index"]; N = int(a["n_nodes"])
for k in (5, 10):
    keep = np.sort(np.concatenate([np.where(EI[1] == d)[0][np.argsort(-vol[EI[1] == d])[:k]] for d in range(N)]))
    b = dict(a); b["fixed_edge_index"] = EI[:, keep]; b["edge_weight"] = a["edge_weight"][:, keep]
    np.savez_compressed(f"graphs_hourly_v2_top{k}.npz", **b); print(k, "->", len(keep), "kenar")
