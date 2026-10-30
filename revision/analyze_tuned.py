import json, numpy as np
from scipy import stats
T={}
for l in open('results_tuned.jsonl'):
    r=json.loads(l); T.setdefault(r['model'],{})[r['seed']]=r['test_mae']
for l in open('results.jsonl'):   # GAT: secilen ayar = varsayilan (64 birim, 1e-3)
    r=json.loads(l)
    if r['model']=='gat' and r['graph']=='real' and r['h']==1 and r['split']=='main' and r['lag']==0:
        T.setdefault('gat',{})[r['seed']]=r['test_mae']
print('Secilen ayarlarla 5 seed (MLP 128/1e-3, GCN 128/5e-3, GAT 64/1e-3), test MAE dk')
for m in ['mlp','gcn','gat']:
    x=np.array([T[m][s] for s in sorted(T[m])]); print(m,'n',len(x),'mean %.3f sd %.3f'%(x.mean(),x.std(ddof=1)),np.round(x,3).tolist())
S=sorted(set(T['mlp'])&set(T['gcn'])&set(T['gat']))
for m in ['gcn','gat']:
    a=np.array([T['mlp'][s] for s in S]); b=np.array([T[m][s] for s in S]); t=stats.ttest_rel(a,b)
    print('mlp-'+m,'diff %.3f'%(a-b).mean(),'p %.4f'%t.pvalue,'MLP',int((a<b).sum()),'/',len(S),'seed daha iyi')
