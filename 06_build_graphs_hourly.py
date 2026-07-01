"""
Adım 1 (saatlik versiyon): BTS 2019 verisinden SAATLİK havalimanı-graf snapshot'ları.

Düğümler  : havalimanları (en yoğun TOP_N_AIRPORTS)
Kenarlar  : o saatteki uçuşlar (Origin -> Dest), ağırlık = uçuş sayısı
Düğüm öznitelikleri (saatlik):
    - kalkan uçuş sayısı
    - ortalama kalkış gecikmesi (DepDelay)
    - inen uçuş sayısı
    - ortalama iniş gecikmesi (ArrDelay)
Hedef:
    - BİR SONRAKİ SAATİN ortalama kalkış gecikmesi (düğüm-seviyesi regresyon)

Saat dilimi, CRSDepTime (HHMM formatı, örn 1430 -> saat 14) sütunundan çıkarılır.
Eğer CRSDepTime yoksa CRS_DEP_TIME denenir.
"""

import pandas as pd
import numpy as np
import pickle
import glob
import os

# ============== AYARLAR ==============
DATA_DIR = "."
YEARS = [2019]
SAMPLE_FRAC = None
TOP_N_AIRPORTS = 50
# =======================================

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
    if not files:
        print(f"UYARI: {pattern} eşleşen dosya bulunamadı, atlanıyor.")
        continue
    for fpath in files:
        print(f"Okunuyor: {fpath}")
        usecols = list(COL_MAP.values())
        df_year = pd.read_csv(fpath, usecols=lambda c: c in usecols, low_memory=False)
        if SAMPLE_FRAC:
            df_year = df_year.sample(frac=SAMPLE_FRAC, random_state=42)
        frames.append(df_year)

if not frames:
    raise FileNotFoundError(f"'{DATA_DIR}' içinde {YEARS} yıllarına ait dosya bulunamadı.")

df = pd.concat(frames, ignore_index=True)
df = df.rename(columns={v: k for k, v in COL_MAP.items()})

if "cancelled" in df.columns:
    df = df[df["cancelled"] == 0]
df = df.dropna(subset=["dep_delay", "arr_delay", "origin", "dest", "crs_dep_time"])

df["date"] = pd.to_datetime(df["date"])

# CRSDepTime: HHMM formatı (örn 1430, 905, 2400->0). Saat dilimini çıkar.
df["crs_dep_time"] = df["crs_dep_time"].astype(int)
df["hour"] = (df["crs_dep_time"] // 100) % 24

# datetime + hour -> tek bir "timeslot" (Timestamp, saat çözünürlüğünde)
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

timeslots = sorted(df["timeslot"].unique())
print(f"Toplam saatlik dilim sayısı: {len(timeslots)}")

# Hızlı erişim için groupby önceden hazırlanır
dep_grouped = df.groupby(["timeslot", "origin"])["dep_delay"].agg(["count", "mean"])
arr_grouped = df.groupby(["timeslot", "dest"])["arr_delay"].agg(["count", "mean"])
edge_grouped = df.groupby(["timeslot", "origin", "dest"]).size()

graphs = []
for ts in timeslots:
    node_feat = np.zeros((n_nodes, 4), dtype=np.float32)

    if ts in dep_grouped.index.get_level_values(0):
        sub = dep_grouped.loc[ts]
        for ap, row in sub.iterrows():
            if ap in airport_to_idx:
                idx = airport_to_idx[ap]
                node_feat[idx, 0] = row["count"]
                node_feat[idx, 1] = row["mean"]

    if ts in arr_grouped.index.get_level_values(0):
        sub = arr_grouped.loc[ts]
        for ap, row in sub.iterrows():
            if ap in airport_to_idx:
                idx = airport_to_idx[ap]
                node_feat[idx, 2] = row["count"]
                node_feat[idx, 3] = row["mean"]

    if ts in edge_grouped.index.get_level_values(0):
        sub = edge_grouped.loc[ts]
        if isinstance(sub, pd.Series):
            pairs = sub.index.tolist()
            counts = sub.values
        else:
            pairs = []
            counts = []
        edge_index = np.array([
            [airport_to_idx[o] for o, d in pairs],
            [airport_to_idx[d] for o, d in pairs],
        ], dtype=np.int64) if pairs else np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.array(counts, dtype=np.float32) if pairs else np.zeros((0,), dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.zeros((0,), dtype=np.float32)

    graphs.append({
        "timeslot": ts,
        "node_feat": node_feat,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
    })

# Hedef: bir sonraki saatin ortalama kalkış gecikmesi
for i in range(len(graphs) - 1):
    graphs[i]["target"] = graphs[i + 1]["node_feat"][:, 1].copy()

graphs = graphs[:-1]

print(f"Kullanılabilir saatlik graf snapshot sayısı: {len(graphs)}")

with open("graphs_hourly.pkl", "wb") as f:
    pickle.dump({"graphs": graphs, "airport_to_idx": airport_to_idx, "n_nodes": n_nodes}, f)

print("graphs_hourly.pkl kaydedildi.")
