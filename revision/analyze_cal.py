import json, numpy as np
from scipy import stats
R={}
for l in open('results_cal.jsonl'):
    r=json.loads(l); R.setdefault(r['model'],{})[r['seed']]=r['test_mae']
S=sorted(R['mlp'])
print('Takvim saati izgarasi (Config 5, 5 seed, test MAE dk)')
for m in ['mlp','gcn','gat']:
    x=np.array([R[m][s] for s in S]); print(m,'mean %.3f sd %.3f'%(x.mean(),x.std(ddof=1)),'seeds',np.round(x,3).tolist())
for m in ['gcn','gat']:
    a=np.array([R['mlp'][s] for s in S]); b=np.array([R[m][s] for s in S]); t=stats.ttest_rel(a,b)
    print('mlp-'+m,'diff %.3f'%(a-b).mean(),'p %.4f'%t.pvalue,'mlp better in',int((a<b).sum()),'of',len(S))
nv=[r for r in json.load(open('naive_cal.json')) if r['h']==1 and r['split']=='main'][0]
print('persistence %.3f  HA %.3f  ridge AR %.3f'%(nv['persistence'],nv['ha'],nv['ar']))
