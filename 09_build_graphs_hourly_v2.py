"""
Adım 9: İyileştirilmiş saatlik graf yapısı.

Önceki versiyondan farklar:
1) SABİT GRAF TOPOLOJİSİ: havalimanları arası bağlantı (kenar varlığı), tüm yıl
   boyunca o rotada en az MIN_EDGE_COUNT uçuş olup olmamasına göre BİR KEZ belirlenir.
   Her zaman adımında AYNI edge_index kullanılır -> GNN tutarlı bir komşuluk yapısı öğrenir.
   Kenar ağırlığı (edge_weight) = o saatteki uçuş sayısı (zamana bağlı, ama yapı sabit).

2) Sadece "aktif" zaman dilimleri tutulur: en az MIN_ACTIVE_NODES havalimanında
   kalkış olan saatler. Tamamen boş/çok seyrek saatler atılır (gürültü azaltma).

3) Hedef: bir sonraki saatin ortalama kalkış gecikmesi. Eğer o saatte o havalimanından
   kalkış yoksa, hedef o havalimanı için NaN işaretlenir ve kayıp hesabında maskelenir
   (önceki versiyonda hedef=0 yapılıyordu, bu yanlış sinyaldi).
"""

import pandas as pd
import numpy as np
import pickle
import glob
import os

DATA_DIR = "."
YEARS = [2019]
TOP_N_AIRPORTS = 50
MIN_EDGE_COUNT = 50      # yıl boyunca en az bu kadar uçuş olan rotalar kenar olarak eklenir
MIN_ACTIVE_NODES = 10    # bir saatlik dilimde en az bu kadar havalimanında kalkış olmalı

COL_MAP = {
    "date": "FlightDate",
    "origin": "Origin",
    "dest": "Dest",
    "dep_delay": "DepDelay",
    "arr_delay": "ArrDelay",
    "cancelled": "Cancelled",
    "crs_dep_time": "CRSDepTime",
}

frames = []
for year in YEARS:
    pattern = os.path.join(DATA_DIR, f"*{year}*.csv")
    files = glob.glob(pattern)
    for fpath in files:
        print(f"Okunuyor: {fpath}")
        usecols = list(COL_MAP.values())
        df_year = pd.read_csv(fpath, usecols=lambda c: c in usecols, low_memory=False)
        frames.append(df_year)

df = pd.concat(frames, ignore_index=True)
df = df.rename(columns={v: k for k, v in COL_MAP.items()})

if "cancelled" in df.columns:
    df = df[df["cancelled"] == 0]
df = df.dropna(subset=["dep_delay", "arr_delay", "origin", "dest", "crs_dep_time"])
df["date"] = pd.to_datetime(df["date"])
df["crs_dep_time"] = df["crs_dep_time"].astype(int)
df["hour"] = (df["crs_dep_time"] // 100) % 24
df["timeslot"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")

print(f"Toplam uçuş kaydı: {len(df):,}")

top_airports = (
    pd.concat([df["origin"], df["dest"]])
    .value_counts()
    .head(TOP_N_AIRPORTS)
    .index
    .tolist()
)
df = df[df["origin"].isin(top_airports) & df["dest"].isin(top_airports)]
print(f"En yoğun {TOP_N_AIRPORTS} havalimanına filtrelendi. Kalan kayıt: {len(df):,}")

all_airports = sorted(set(df["origin"]) | set(df["dest"]))
airport_to_idx = {a: i for i, a in enumerate(all_airports)}
n_nodes = len(all_airports)
print(f"Düğüm (havalimanı) sayısı: {n_nodes}")

# ---- SABİT GRAF TOPOLOJİSİ ----
route_counts = df.groupby(["origin", "dest"]).size().reset_index(name="count")
route_counts = route_counts[route_counts["count"] >= MIN_EDGE_COUNT]
fixed_edges = list(zip(route_counts["origin"], route_counts["dest"]))
fixed_edge_index = np.array([
    [airport_to_idx[o] for o, d in fixed_edges],
    [airport_to_idx[d] for o, d in fixed_edges],
], dtype=np.int64)
print(f"Sabit graf kenar sayısı: {fixed_edge_index.shape[1]} (eşik: >= {MIN_EDGE_COUNT} uçuş/yıl)")

edge_key_to_pos = {(o, d): i for i, (o, d) in enumerate(fixed_edges)}

# ---- ZAMAN DİLİMLERİ ----
timeslots = sorted(df["timeslot"].unique())
print(f"Toplam saatlik dilim sayısı: {len(timeslots)}")

dep_grouped = df.groupby(["timeslot", "origin"])["dep_delay"].agg(["count", "mean"])
arr_grouped = df.groupby(["timeslot", "dest"])["arr_delay"].agg(["count", "mean"])
edge_grouped = df.groupby(["timeslot", "origin", "dest"]).size()

graphs = []
for ts in timeslots:
    node_feat = np.zeros((n_nodes, 4), dtype=np.float32)
    has_dep = np.zeros(n_nodes, dtype=bool)

    if ts in dep_grouped.index.get_level_values(0):
        sub = dep_grouped.loc[ts]
        for ap, row in sub.iterrows():
            if ap in airport_to_idx:
                idx = airport_to_idx[ap]
                node_feat[idx, 0] = row["count"]
                node_feat[idx, 1] = row["mean"]
                has_dep[idx] = True

    if ts in arr_grouped.index.get_level_values(0):
        sub = arr_grouped.loc[ts]
        for ap, row in sub.iterrows():
            if ap in airport_to_idx:
                idx = airport_to_idx[ap]
                node_feat[idx, 2] = row["count"]
                node_feat[idx, 3] = row["mean"]

    if has_dep.sum() < MIN_ACTIVE_NODES:
        continue  # çok seyrek saat, atla

    # Sabit topoloji üzerinde kenar ağırlıkları (o saatteki uçuş sayısı, yoksa 0)
    edge_weight = np.zeros(fixed_edge_index.shape[1], dtype=np.float32)
    if ts in edge_grouped.index.get_level_values(0):
        sub = edge_grouped.loc[ts]
        if isinstance(sub, pd.Series):
            for (o, d), cnt in sub.items():
                key = (o, d)
                if key in edge_key_to_pos:
                    edge_weight[edge_key_to_pos[key]] = cnt

    graphs.append({
        "timeslot": ts,
        "node_feat": node_feat,
        "edge_index": fixed_edge_index,
        "edge_weight": edge_weight,
        "has_dep": has_dep,
    })

print(f"Aktif (>= {MIN_ACTIVE_NODES} havalimanı) saatlik dilim sayısı: {len(graphs)}")

# Hedef: bir sonraki AKTİF zaman diliminin ortalama kalkış gecikmesi.
# Eğer hedef havalimanından o dilimde kalkış yoksa -> NaN (maskelenecek)
for i in range(len(graphs) - 1):
    next_feat = graphs[i + 1]["node_feat"][:, 1].copy()
    next_has_dep = graphs[i + 1]["has_dep"]
    target = np.where(next_has_dep, next_feat, np.nan).astype(np.float32)
    graphs[i]["target"] = target
    graphs[i]["target_mask"] = next_has_dep.astype(np.float32)

graphs = graphs[:-1]

print(f"Kullanılabilir graf snapshot sayısı: {len(graphs)}")
avg_valid = np.mean([g["target_mask"].sum() for g in graphs])
print(f"Snapshot başına ortalama geçerli hedef düğüm sayısı: {avg_valid:.1f} / {n_nodes}")

with open("graphs_hourly_v2.pkl", "wb") as f:
    pickle.dump({"graphs": graphs, "airport_to_idx": airport_to_idx, "n_nodes": n_nodes,
                  "fixed_edge_index": fixed_edge_index}, f)

print("graphs_hourly_v2.pkl kaydedildi.")
