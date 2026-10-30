import json, os, numpy as np
from scipy import stats
def load(f, cond=lambda r: True):
    D={}
    if not os.path.exists(f): return D
    for l in open(f):
        r=json.loads(l)
        if cond(r): D.setdefault((r['model'],r['data']),{})[r['seed']]=r['test_mae']
    return D
main=load('results.jsonl', lambda r: r['h']==1 and r['split']=='main' and r['lag']==0 and r['graph']=='real')
ref={m:v for (m,_),v in main.items()}
def paired(a,b):
    s=sorted(set(a)&set(b)); x=np.array([a[k] for k in s]); y=np.array([b[k] for k in s])
    return (x-y).mean(), (stats.ttest_rel(x,y).pvalue if len(s)>1 else np.nan), int((x<y).sum()), len(s)
for title,f in [('SEYREK GRAF (her havalimani icin en yogun k gelen rota)','results_sparse.jsonl'),('SAATLIK HAVA DURUMU','results_weather.jsonl')]:
    D=load(f, lambda r: r['graph']=='real')
    if not D: print('==',title,': sonuc yok'); continue
    print('==',title)
    for (m,data),v in sorted(D.items()):
        x=np.array(list(v.values()))
        line=f'{m:6s} {data:30s} n {len(x)} mean {x.mean():.3f} sd {x.std(ddof=1) if len(x)>1 else 0:.3f}'
        if m in ref:
            d,p,w,n=paired(v,ref[m]); line+=f' | vs ayni model yogun/havasiz: {d:+.3f} p {p:.4f} ({w}/{n} seed daha iyi)'
        if 'mlp' in ref and m!='mlp':
            d,p,w,n=paired(v,ref['mlp']); line+=f' | vs MLP (ana): {d:+.3f} p {p:.4f}'
        print(line)
    W=D.get(('mlp','graphs_hourly_v2_weather.npz'))
    if W:
        for m in ['gcn','gat']:
            v=D.get((m,'graphs_hourly_v2_weather.npz'))
            if v: d,p,w,n=paired(v,W); print(f'{m} vs MLP (ikisi de hava durumlu): {d:+.3f} p {p:.4f} (MLP {n-w}/{n} seed daha iyi)')
