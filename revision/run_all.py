import subprocess, sys, os
done=set(open('done.txt').read().splitlines()) if os.path.exists('done.txt') else set()
for line in open('queue.txt').read().splitlines():
    if not line or line in done: continue
    print('>>',line,flush=True)
    r=subprocess.run([sys.executable,'exp.py']+line.split())
    if r.returncode==0: open('done.txt','a').write(line+'\n')
print('Bitti. Sonra: python analyze.py > analiz.txt')
