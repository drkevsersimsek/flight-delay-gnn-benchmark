import subprocess, sys, os
q=sys.argv[1]; df=q.replace('.txt','_done.txt')
done=set(open(df).read().splitlines()) if os.path.exists(df) else set()
for line in open(q).read().splitlines():
    if not line or line in done: continue
    print('>>',line,flush=True)
    r=subprocess.run([sys.executable,'exp.py']+line.split())
    if r.returncode==0: open(df,'a').write(line+'\n')
print('Bitti:',q)
