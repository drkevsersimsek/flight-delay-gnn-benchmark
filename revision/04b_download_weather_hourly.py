"""Hakem 2, madde 5: 50 havalimani icin 2019 saatlik ASOS gozlemlerini IEM'den indirir.
Cikti: weather_hourly/XXX.csv  (yerel saatle)"""
import os, time, urllib.request
TZ = {'ATL':'America/New_York','AUS':'America/Chicago','BNA':'America/Chicago','BOS':'America/New_York','BWI':'America/New_York',
'CLE':'America/New_York','CLT':'America/New_York','CMH':'America/New_York','CVG':'America/New_York','DAL':'America/Chicago',
'DCA':'America/New_York','DEN':'America/Denver','DFW':'America/Chicago','DTW':'America/Detroit','EWR':'America/New_York',
'FLL':'America/New_York','HNL':'Pacific/Honolulu','HOU':'America/Chicago','IAD':'America/New_York','IAH':'America/Chicago',
'IND':'America/Indiana/Indianapolis','JAX':'America/New_York','JFK':'America/New_York','LAS':'America/Los_Angeles',
'LAX':'America/Los_Angeles','LGA':'America/New_York','MCI':'America/Chicago','MCO':'America/New_York','MDW':'America/Chicago',
'MIA':'America/New_York','MKE':'America/Chicago','MSP':'America/Chicago','MSY':'America/Chicago','OAK':'America/Los_Angeles',
'ORD':'America/Chicago','PDX':'America/Los_Angeles','PHL':'America/New_York','PHX':'America/Phoenix','PIT':'America/New_York',
'RDU':'America/New_York','SAN':'America/Los_Angeles','SAT':'America/Chicago','SEA':'America/Los_Angeles','SFO':'America/Los_Angeles',
'SJC':'America/Los_Angeles','SLC':'America/Denver','SMF':'America/Los_Angeles','SNA':'America/Los_Angeles','STL':'America/Chicago',
'TPA':'America/New_York'}
BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
os.makedirs("weather_hourly", exist_ok=True)
failed = []
for ap, tz in TZ.items():
    out = os.path.join("weather_hourly", f"{ap}.csv")
    if os.path.exists(out) and os.path.getsize(out) > 10000:
        print(ap, "zaten var, atlaniyor"); continue
    url = (f"{BASE}?station={ap}&data=tmpf&data=sknt&data=p01i&data=vsby"
           f"&year1=2019&month1=1&day1=1&year2=2020&month2=1&day2=2"
           f"&tz={tz}&format=onlycomma&latlon=no&missing=M&trace=0.0001&direct=no&report_type=3&report_type=4")
    for attempt in range(5):
        try:
            data = urllib.request.urlopen(url, timeout=300).read()
            open(out, "wb").write(data)
            print(ap, "indirildi,", data.count(b"\n"), "satir"); break
        except Exception as e:
            print(ap, "hata:", e, "- tekrar deneniyor"); time.sleep(10 * (attempt + 1))
    else:
        failed.append(ap)
    time.sleep(3)
print("Bitti." if not failed else f"Indirilemeyenler: {failed} - betigi tekrar calistirin")
