import json, glob, numpy as np, collections
from scipy import stats
R=[json.loads(l) for l in open('results.jsonl')]
g=collections.defaultdict(dict)
for r in R:
    k=(r['model'],r['graph'],r['h'],r['split'],r['lag']); g[k][r['seed']]=r
def summ(k):
    v=g.get(k,{}); x=np.array([v[s]['test_mae'] for s in sorted(v)])
    return x
print('== main h1')
base=g[('mlp','real',1,'main',0)]
for k in sorted(g):
    if k[2]==1 and k[3]=='main' and k[4]==0:
        x=summ(k); p=list(g[k].values())[0]['params']
        seeds=sorted(set(g[k])&set(base)); d=np.array([g[k][s]['test_mae']-base[s]['test_mae'] for s in seeds])
        t=stats.ttest_rel([g[k][s]['test_mae'] for s in seeds],[base[s]['test_mae'] for s in seeds]) if len(seeds)>1 and k[0]!='mlp' else None
        print(k[0],k[1],'n',len(x),'mean %.3f sd %.3f'%(x.mean(),x.std(ddof=1) if len(x)>1 else 0),'d %.3f'%d.mean(),'wins_mlp',(d>0).sum(),'p',None if t is None else round(t.pvalue,4),'params',p, 'eps',[g[k][s]['best_epoch'] for s in sorted(g[k])])
print('== real minus random / identity (paired by seed)')
for m in ['gcn','gat','sage']:
    for gr in ['random','identity']:
        a=g.get((m,'real',1,'main',0),{}); b=g.get((m,gr,1,'main',0),{}); s=sorted(set(a)&set(b))
        if len(s)>1:
            d=np.array([a[x]['test_mae']-b[x]['test_mae'] for x in s]); t=stats.ttest_rel([a[x]['test_mae'] for x in s],[b[x]['test_mae'] for x in s])
            print(m,'real-'+gr,'%.3f'%d.mean(),'sd %.3f'%d.std(ddof=1),'p %.4f'%t.pvalue,'n',len(s))
print('== horizon / rolling / lag')
for k in sorted(g):
    if k[2]!=1 or k[3]!='main' or k[4]!=0:
        x=summ(k); print(k,'n',len(x),'mean %.3f sd %.3f'%(x.mean(),x.std(ddof=1) if len(x)>1 else 0))
print('== disruption (top 10% network-mean delay test snapshots), h1 main')
nv=np.load('naive_preds.npz')
yt,mk=nv['yt'],nv['mk']; net=np.array([yt[i][mk[i]].mean() if mk[i].any() else -1e9 for i in range(len(yt))]); top=net>=np.quantile(net,0.9)
def dm(pr,yt,mk,sel): m=mk[sel]>0; return np.abs(pr[sel]-yt[sel])[m].mean()
print('persistence all %.3f top %.3f'%(dm(nv['pers'],yt,mk,slice(None)),dm(nv['pers'],yt,mk,top)))
print('AR all %.3f top %.3f'%(dm(nv['ar'],yt,mk,slice(None)),dm(nv['ar'],yt,mk,top)))
for m in ['mlp','gcn','gat','sage','mlpnb']:
    fs=sorted(glob.glob(f'preds/{m}_real_train_h1_main_lag0_H64_*'))
    if fs:
        v=[]
        for f in fs: z=np.load(f); v.append((dm(z['pr'],z['yt'],z['mk'],slice(None)),dm(z['pr'],z['yt'],z['mk'],top)))
        v=np.array(v); print(m,'n',len(v),'all %.3f top %.3f sd %.3f'%(v[:,0].mean(),v[:,1].mean(),v[:,1].std(ddof=1) if len(v)>1 else 0))
print('top10 threshold network-mean delay (min): %.1f'%np.quantile(net,0.9))
