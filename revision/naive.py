import pickle, json, numpy as np, pandas as pd
import sys
DATA=sys.argv[1] if len(sys.argv)>1 else 'graphs_hourly_v2.npz'; SUF=sys.argv[2] if len(sys.argv)>2 else ''
from loadgraphs import load_graphs; o=load_graphs(DATA); G=o['graphs']; N=o['n_nodes']; W=24; T=len(G)
X=np.stack([g['node_feat'] for g in G]); HD=np.stack([g['has_dep'] for g in G])
Yr=np.nan_to_num(np.stack([g['target'] for g in G])); Mr=np.stack([g['target_mask'] for g in G])>0
TS=pd.to_datetime([g['timeslot'] for g in G]); TSn=TS.values
last=np.zeros((T,N)); cur=np.zeros(N)
for t in range(T): cur=np.where(HD[t],X[t,:,1],cur); last[t]=cur
def splits(h,sp):
    idx=np.arange(W-1,T-(h-1))
    if sp=='main': s=int(len(idx)*.8); return idx[:s],idx[s:]
    q={'rollQ2':('2019-04-01','2019-07-01'),'rollQ3':('2019-07-01','2019-10-01'),'rollQ4':('2019-10-01','2020-01-01')}[sp]
    t0,t1=np.datetime64(q[0]),np.datetime64(q[1]); return idx[TSn[idx]<t0],idx[(TSn[idx]>=t0)&(TSn[idx]<t1)]
out=[]
for h,sp in [(h,'main') for h in range(1,7)]+[(1,s) for s in ['rollQ2','rollQ3','rollQ4']]:
    tr,te=splits(h,sp); v=int(len(tr)*.8); itr,va=tr[:v],tr[v:]
    K=lambda ids: ids+h-1
    def mae(P,ids): m=Mr[K(ids)]; return float(np.abs(P-Yr[K(ids)])[m].mean())
    res={'h':h,'split':sp,'persistence':mae(last[te],te)}
    # HA by airport,hour,weekday of target slot (slot of snapshot k+1)
    tsl=np.r_[TSn[1:],TSn[-1:]]; hh=pd.DatetimeIndex(tsl).hour.values; wd=pd.DatetimeIndex(tsl).dayofweek.values
    k=K(tr); m=Mr[k]; nodes=np.broadcast_to(np.arange(N),m.shape)
    df=pd.DataFrame({'n':nodes[m],'h':np.repeat(hh[k],N).reshape(m.shape)[m],'w':np.repeat(wd[k],N).reshape(m.shape)[m],'y':Yr[k][m]})
    tab=df.groupby(['n','h','w']).y.mean(); nm=df.groupby('n').y.mean()
    full=np.zeros((N,24,7))
    for a_ in range(N): full[a_]=nm.get(a_,0)
    for (a_,h_,w_),val in tab.items(): full[a_,h_,w_]=val
    kt=K(te); res['ha']=mae(full[np.arange(N)[None],hh[kt][:,None],wd[kt][:,None]],te)
    mu=X[:tr[-1]+1].reshape(-1,4).mean(0); sd=X[:tr[-1]+1].reshape(-1,4).std(0)+1e-6; Z=(X-mu)/sd
    def ds(ids):
        F=np.concatenate([Z[ids-W+1+j] for j in range(W)],axis=2); m=Mr[K(ids)]; return F[m],Yr[K(ids)][m]
    A,b=ds(itr); A1=np.c_[A,np.ones(len(A))]; Fv,yv=ds(va); Fv=np.c_[Fv,np.ones(len(Fv))]
    best=None
    for al in [1e3,1e4,3e4,1e5,3e5,1e6]:
        w=np.linalg.solve(A1.T@A1+al*np.eye(A1.shape[1]),A1.T@b); e=np.abs(Fv@w-yv).mean()
        if best is None or e<best[0]: best=(e,al,w)
    Ft,yt=ds(te); res['ar']=float(np.abs(np.c_[Ft,np.ones(len(Ft))]@best[2]-yt).mean()); res['ar_alpha']=best[1]
    if h==1 and sp=='main':  # save per-sample preds for disruption analysis
        F=np.concatenate([Z[te-W+1+j] for j in range(W)],axis=2); P=np.c_[F.reshape(-1,96),np.ones(len(te)*N)]@best[2]
        np.savez(f'naive_preds{SUF}.npz',sid=te,ar=P.reshape(len(te),N),pers=last[te],yt=Yr[K(te)],mk=Mr[K(te)])
    out.append(res); print(res,flush=True)
json.dump(out,open(f'naive{SUF}.json','w'))
