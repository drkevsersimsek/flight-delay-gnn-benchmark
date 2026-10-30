import subprocess, sys, os
done=set(open('done_cal.txt').read().splitlines()) if os.path.exists('done_cal.txt') else set()
for line in open('queue_cal.txt').read().splitlines():
    if not line or line in done: continue
    print('>>',line,flush=True)
    r=subprocess.run([sys.executable,'exp.py']+line.split())
    if r.returncode==0: open('done_cal.txt','a').write(line+'\n')
print('Bitti. Sonra: python analyze_cal.py > analiz_cal.txt')
