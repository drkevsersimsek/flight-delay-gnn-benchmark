"""
Adım 1: BTS Flight Delay Dataset (2018-2022, Kaggle: robikscube/flight-delay-dataset-20182022)
verisinden günlük (veya saatlik) havalimanı-graf snapshot'ları oluşturma.

KULLANIM:
1) Kaggle'dan veriyi indirin:
   https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022
   (kaggle datasets download -d robikscube/flight-delay-dataset-20182022)

2) DATA_DIR değişkenini indirdiğiniz klasörün yolu ile değiştirin.
   Veri seti genelde "Combined_Flights_2018.csv" ... "Combined_Flights_2022.csv"
   şeklinde yıllık dosyalar halinde gelir. YEARS listesini ihtiyacınıza göre ayarlayın.

3) (Opsiyonel) Tüm yıllar = ~29M satır, RAM'e sığmayabilir. Bu yüzden:
   - SAMPLE_FRAC ile örnekleme yapılabilir
   - veya tek yıl (örn. 2019) ile başlanması önerilir

Düğümler  : havalimanları (Origin + Dest kümesi, IATA kodu)
Kenarlar  : o gündeki uçuşlar (Origin -> Dest), ağırlık = uçuş sayısı
Düğüm öznitelikleri (günlük):
    - kalkan uçuş sayısı
    - ortalama kalkış gecikmesi (DepDelay)
    - inen uçuş sayısı
    - ortalama iniş gecikmesi (ArrDelay)
Hedef:
    - bir sonraki günün ortalama kalkış gecikmesi (düğüm-seviyesi regresyon)
"""

import pandas as pd
import numpy as np
import pickle
import glob
import os

# ============== AYARLAR (kendi ortamınıza göre düzenleyin) ==============
DATA_DIR = "."   # Kaggle'dan indirilen klasör
YEARS = [2019]                          # tek yılla başlamak için. örn: [2018, 2019] de olabilir
SAMPLE_FRAC = None                      # örn. 0.2 -> verinin %20'sini rastgele al (RAM kısıtı varsa)
TOP_N_AIRPORTS = 50                     # çok küçük havalimanlarını eleyip en yoğun N havalimanıyla sınırla
# ==========================================================================

# BTS / Kaggle Combined_Flights veri setindeki olası sütun adları
# (Kaggle versiyonuna göre büyük/küçük harf değişebilir, gerekirse düzeltin)
COL_MAP = {
    "date": "FlightDate",
    "origin": "Origin",
    "dest": "Dest",
    "dep_delay": "DepDelay",
    "arr_delay": "ArrDelay",
    "cancelled": "Cancelled",
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
    raise FileNotFoundError(
        f"'{DATA_DIR}' içinde {YEARS} yıllarına ait dosya bulunamadı. "
        "DATA_DIR ve YEARS değişkenlerini kontrol edin."
    )

df = pd.concat(frames, ignore_index=True)
df = df.rename(columns={v: k for k, v in COL_MAP.items()})

# Temizlik: iptal edilen uçuşları ve eksik gecikme verilerini çıkar
if "cancelled" in df.columns:
    df = df[df["cancelled"] == 0]
df = df.dropna(subset=["dep_delay", "arr_delay", "origin", "dest"])

df["date"] = pd.to_datetime(df["date"])

print(f"Toplam uçuş kaydı: {len(df):,}")

# En yoğun N havalimanı ile sınırla (yoksa graf çok seyrek/büyük olur)
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

dates = sorted(df["date"].unique())
print(f"Gün sayısı: {len(dates)}")

graphs = []
for date in dates:
    day_df = df[df["date"] == date]

    node_feat = np.zeros((n_nodes, 4), dtype=np.float32)

    dep_stats = day_df.groupby("origin")["dep_delay"].agg(["count", "mean"])
    for ap, row in dep_stats.iterrows():
        idx = airport_to_idx[ap]
        node_feat[idx, 0] = row["count"]
        node_feat[idx, 1] = row["mean"]

    arr_stats = day_df.groupby("dest")["arr_delay"].agg(["count", "mean"])
    for ap, row in arr_stats.iterrows():
        idx = airport_to_idx[ap]
        node_feat[idx, 2] = row["count"]
        node_feat[idx, 3] = row["mean"]

    edge_counts = day_df.groupby(["origin", "dest"]).size().reset_index(name="count")
    edge_index = np.array([
        [airport_to_idx[o] for o in edge_counts["origin"]],
        [airport_to_idx[d] for d in edge_counts["dest"]],
    ], dtype=np.int64)
    edge_weight = edge_counts["count"].values.astype(np.float32)

    graphs.append({
        "date": date,
        "node_feat": node_feat,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
    })

for i in range(len(graphs) - 1):
    graphs[i]["target"] = graphs[i + 1]["node_feat"][:, 1].copy()

graphs = graphs[:-1]

print(f"Kullanılabilir graf snapshot sayısı: {len(graphs)}")

with open("graphs.pkl", "wb") as f:
    pickle.dump({"graphs": graphs, "airport_to_idx": airport_to_idx, "n_nodes": n_nodes}, f)

print("graphs.pkl kaydedildi.")
