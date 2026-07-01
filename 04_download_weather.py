"""
Adım Ek-1 (düzeltilmiş): Iowa State ASOS arşivinden 2019 günlük hava durumu verisi indirme.

Yöntem:
1) ABD eyalet bazlı ASOS network'lerinin (XX_ASOS) istasyon listesini geojson'dan çek,
   her IATA koduna ait doğru network'ü bul.
2) daily.py endpoint'ini doğru network + sts/ets parametreleriyle çağır.

Çıktı: weather_2019.csv  (sütunlar: station, day, max_temp_f, min_temp_f, precip_in, avg_wind_speed_kts)

Not: Bu script internet bağlantısı gerektirir, kendi bilgisayarınızda çalıştırın.
İlk çalıştırmada network-istasyon haritası çekileceği için biraz sürebilir (~1-2 dk).
"""

import urllib.request
import json
import pickle
import time
import csv

with open("graphs.pkl", "rb") as f:
    obj = pickle.load(f)

stations = sorted(obj["airport_to_idx"].keys())
print(f"{len(stations)} havalimanı için hava durumu indirilecek.")

STATES = (
    "AK AL AR AZ CA CO CT DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN "
    "MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT "
    "WA WI WV WY"
).split()

print("Eyalet ASOS network'lerinden istasyon listesi çekiliyor...")
station_to_network = {}
for state in STATES:
    network = f"{state}_ASOS"
    url = f"https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            jdict = json.load(resp)
        for feat in jdict["features"]:
            sid = feat["properties"]["sid"]
            station_to_network[sid] = network
    except Exception as e:
        print(f"  UYARI: {network} alınamadı ({e})")
    time.sleep(0.1)

print(f"Toplam {len(station_to_network)} istasyon-network eşlemesi bulundu.")

BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"

all_rows = []
header = None


def fetch_station(station_code, network):
    params = (
        f"network={network}&stations={station_code}"
        f"&sts=2019-01-01&ets=2019-12-31"
        f"&var=max_temp_f&var=min_temp_f&var=precip_in&var=avg_wind_speed_kts"
        f"&na=blank&format=csv"
    )
    url = f"{BASE_URL}?{params}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
    return text


for i, st in enumerate(stations, 1):
    # IATA -> ICAO eşlemesi genelde 'K' + IATA (3 harfli kodlar için)
    candidates = []
    if st in station_to_network:
        candidates.append(st)
    icao = "K" + st
    if icao in station_to_network:
        candidates.append(icao)

    if not candidates:
        print(f"  [{i}/{len(stations)}] {st}: network bulunamadı, atlanıyor.")
        continue

    found = False
    for code in candidates:
        network = station_to_network[code]
        try:
            text = fetch_station(code, network)
        except Exception as e:
            print(f"  [{i}/{len(stations)}] {st} ({code}/{network}): hata ({e})")
            continue

        lines = text.strip().splitlines() if text else []
        if len(lines) <= 1:
            continue

        reader = csv.reader(lines)
        rows = list(reader)
        if header is None:
            header = rows[0]
        data_rows = rows[1:]

        # station sütununu IATA koduyla normalize et
        for r in data_rows:
            if r and r[0] == icao:
                r[0] = st
            all_rows.append(r)

        print(f"  [{i}/{len(stations)}] {st} ({code}/{network}): {len(data_rows)} satır indirildi.")
        found = True
        break

    if not found:
        print(f"  [{i}/{len(stations)}] {st}: veri bulunamadı, atlanıyor.")

    time.sleep(0.3)

if header is None:
    raise RuntimeError("Hiçbir istasyon için veri indirilemedi.")

with open("weather_2019.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all_rows)

print(f"\nweather_2019.csv kaydedildi. Toplam satır: {len(all_rows)}")
