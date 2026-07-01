"""
Adım Ek-2: weather_2019.csv verisini graphs.pkl'deki günlük graf snapshot'larına
ekleyerek graphs_weather.pkl üretir.

Eklenen düğüm öznitelikleri (mevcut 4'e ek olarak):
    - max_temp_f
    - precip_in
    - avg_wind_speed_kts

Yani her düğümün öznitelik vektörü 4 -> 7 boyuta çıkar.
Eksik hava durumu verisi olan gün/havalimanı kombinasyonları için 0 kullanılır.
"""

import pickle
import numpy as np
import pandas as pd

with open("graphs.pkl", "rb") as f:
    obj = pickle.load(f)

graphs = obj["graphs"]
airport_to_idx = obj["airport_to_idx"]
n_nodes = obj["n_nodes"]

weather = pd.read_csv("weather_2019.csv")
print("weather_2019.csv sütunları:", list(weather.columns))

# Sütun adlarını normalize et (IEM çıktısında 'station' ve 'day' olmalı)
weather.columns = [c.strip().lower() for c in weather.columns]
weather["day"] = pd.to_datetime(weather["day"])

# Sayısal sütunları zorla numeric yap (boşlar NaN olur -> 0 ile doldurulacak)
for col in ["max_temp_f", "precip_in", "avg_wind_speed_kts"]:
    if col not in weather.columns:
        weather[col] = np.nan
    weather[col] = pd.to_numeric(weather[col], errors="coerce")

# (station, day) -> (max_temp_f, precip_in, avg_wind_speed_kts) sözlüğü
weather_lookup = {}
for _, row in weather.iterrows():
    key = (row["station"], row["day"])
    weather_lookup[key] = (
        row["max_temp_f"] if not np.isnan(row["max_temp_f"]) else 0.0,
        row["precip_in"] if not np.isnan(row["precip_in"]) else 0.0,
        row["avg_wind_speed_kts"] if not np.isnan(row["avg_wind_speed_kts"]) else 0.0,
    )

print(f"Hava durumu kayıt sayısı: {len(weather_lookup)}")

matched, missing = 0, 0
for g in graphs:
    date = pd.Timestamp(g["date"])
    old_feat = g["node_feat"]  # [n_nodes, 4]
    weather_feat = np.zeros((n_nodes, 3), dtype=np.float32)

    for ap, idx in airport_to_idx.items():
        key = (ap, date)
        if key in weather_lookup:
            weather_feat[idx, :] = weather_lookup[key]
            matched += 1
        else:
            missing += 1

    g["node_feat"] = np.concatenate([old_feat, weather_feat], axis=1)  # [n_nodes, 7]

print(f"Eşleşen (havalimanı, gün) çifti: {matched}")
print(f"Eksik (0 ile doldurulan): {missing}")

# Hedef değişkeni de eski 4 öznitelik üzerinden hesaplanmış olabilir, kontrol edelim:
print("Yeni öznitelik boyutu:", graphs[0]["node_feat"].shape)
print("Hedef boyutu:", graphs[0]["target"].shape)

with open("graphs_weather.pkl", "wb") as f:
    pickle.dump({"graphs": graphs, "airport_to_idx": airport_to_idx, "n_nodes": n_nodes}, f)

print("graphs_weather.pkl kaydedildi.")
