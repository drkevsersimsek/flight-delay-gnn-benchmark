"""HNL icin saatlik ASOS verisini IEM'den 'PHNL' kodu ile indirir (HNL kodu bos donuyor)."""
import urllib.request
for st in ["PHNL", "HNL"]:
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=" + st +
           "&data=tmpf&data=sknt&data=p01i&data=vsby&year1=2019&month1=1&day1=1&year2=2020&month2=1&day2=2"
           "&tz=Pacific/Honolulu&format=onlycomma&latlon=no&missing=M&trace=0.0001&direct=no&report_type=3&report_type=4")
    data = urllib.request.urlopen(url, timeout=300).read()
    n = data.count(b"\n")
    print(st, "->", n, "satir")
    if n > 1000:
        open("weather_hourly/HNL.csv", "wb").write(data)
        print("weather_hourly/HNL.csv kaydedildi."); break
else:
    print("HNL verisi bulunamadi; HNL eksik olarak kalacak (gostergeyle isaretli).")
