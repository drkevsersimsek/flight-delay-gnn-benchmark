"""Saatlik hava durumunu Config 5 snapshot'larina ekler -> graphs_hourly_v2_weather.npz
Her saat icin 5 yeni ozellik: sicaklik (ort.), ruzgar (ort.), yagis (maks.), gorus (min.), gozlem-yok gostergesi.
Eksik degerler, standardizasyondan ONCE havalimaninin egitim donemi ortalamasiyla doldurulur (HNL hatasi tekrarlanmaz)."""
import os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
z = np.load("graphs_hourly_v2.npz"); a = {k: z[k] for k in z.files}
AP = ['ATL','AUS','BNA','BOS','BWI','CLE','CLT','CMH','CVG','DAL','DCA','DEN','DFW','DTW','EWR','FLL','HNL','HOU','IAD','IAH',
      'IND','JAX','JFK','LAS','LAX','LGA','MCI','MCO','MDW','MIA','MKE','MSP','MSY','OAK','ORD','PDX','PHL','PHX','PIT','RDU',
      'SAN','SAT','SEA','SFO','SJC','SLC','SMF','SNA','STL','TPA']   # graphs_hourly_v2 dugum sirasi
ts = pd.to_datetime(a["timeslot"].astype("datetime64[ns]")); T = len(ts); N = len(AP)
W = 24; n = T - W + 1; last_train = W - 1 + int(n * 0.8) - 1
VARS = ["tmpf", "sknt", "p01i", "vsby"]
Wx = np.full((T, N, 4), np.nan, np.float32)
for j, ap in enumerate(AP):
    f = os.path.join("weather_hourly", f"{ap}.csv")
    if not os.path.exists(f): print(ap, "dosya yok -> tamamen eksik"); continue
    d = pd.read_csv(f, na_values=["M"], comment="#")
    if d.empty or "valid" not in d: print(ap, "bos dosya"); continue
    d["hour"] = pd.to_datetime(d["valid"]).dt.floor("h")
    for c in VARS: d[c] = pd.to_numeric(d[c], errors="coerce")
    g = d.groupby("hour").agg(tmpf=("tmpf","mean"), sknt=("sknt","mean"), p01i=("p01i","max"), vsby=("vsby","min"))
    g.loc[g['p01i'].isna(), 'p01i'] = 0.0      # yagis raporu yoksa yagis yok
    g = g.reindex(ts)
    Wx[:, j, :] = g[VARS].values
    print(f"{ap}: saatlerin %{100*np.isnan(Wx[:,j,0]).mean():.1f}'inde sicaklik eksik")
miss = np.isnan(Wx[:, :, :1]).astype(np.float32)   # o saatte hic gozlem yok (sicaklik eksik)
for j in range(N):
    for v in range(4):
        col = Wx[:last_train+1, j, v]
        m = np.nanmean(col) if np.isfinite(col).any() else np.nanmean(Wx[:last_train+1, :, v])
        Wx[np.isnan(Wx[:, j, v]), j, v] = m if np.isfinite(m) else 0.0
a["node_feat"] = np.concatenate([a["node_feat"], Wx, miss], axis=2).astype(np.float32)
np.savez_compressed("graphs_hourly_v2_weather.npz", **a)
print("Toplam eksik gozlem orani: %.1f%%" % (100 * miss.mean()))
print("graphs_hourly_v2_weather.npz kaydedildi; ozellik sayisi:", a["node_feat"].shape[2])
