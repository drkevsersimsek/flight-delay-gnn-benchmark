import argparse, json, pickle, time, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
torch.set_num_threads(1)
p=argparse.ArgumentParser()
p.add_argument('--model'); p.add_argument('--graph',default='real'); p.add_argument('--seed',type=int,default=42)
p.add_argument('--norm',default='train'); p.add_argument('--h',type=int,default=1); p.add_argument('--hidden',type=int,default=64)
p.add_argument('--lr',type=float,default=None); p.add_argument('--dropout',type=float,default=0.2)
p.add_argument('--epochs',type=int,default=80); p.add_argument('--patience',type=int,default=15)
p.add_argument('--split',default='main')  # main | rollQ2 | rollQ3 | rollQ4
p.add_argument('--lag',type=int,default=0)
p.add_argument('--tag',default=''); p.add_argument('--out',default='results.jsonl')
p.add_argument('--data',default='graphs_hourly_v2.npz'); p.add_argument('--preds',default='preds')
a=p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
from loadgraphs import load_graphs; o=load_graphs(a.data); G=o['graphs']; N=o['n_nodes']; EI=o['fixed_edge_index']
W=24; T=len(G)
X=np.stack([g['node_feat'] for g in G]).astype(np.float32)
EW=np.stack([g['edge_weight'] for g in G]).astype(np.float32)
# target for sample i at horizon h = dep delay at snapshot i+h ; G[i+h-1]['target']
Yr=np.stack([g['target'] for g in G]); Mr=np.stack([g['target_mask'] for g in G])
NF=X.shape[2]
TS=np.array([np.datetime64(g['timeslot']) for g in G])
if a.lag:  # lag delay features: delay columns of snapshot t replaced by those of t-1
    X2=X.copy(); X2[1:,:,1]=X[:-1,:,1]; X2[1:,:,3]=X[:-1,:,3]; X2[0,:,1]=0; X2[0,:,3]=0; X=X2
idx=np.arange(W-1,T-(a.h-1))
if a.split=='main':
    n=len(idx); s=int(n*.8); tr,te=idx[:s],idx[s:]
else:
    q={'rollQ2':('2019-04-01','2019-07-01'),'rollQ3':('2019-07-01','2019-10-01'),'rollQ4':('2019-10-01','2020-01-01')}[a.split]
    t0,t1=np.datetime64(q[0]),np.datetime64(q[1])
    tr=idx[TS[idx]<t0]; te=idx[(TS[idx]>=t0)&(TS[idx]<t1)]
v=int(len(tr)*.8); itr,va=tr[:v],tr[v:]
# drop train samples whose target lies in test region (horizon>1 boundary)
fit_last = tr[-1]
if a.norm=='train':
    fx=X[:fit_last+1].reshape(-1,NF); ew=np.log1p(EW[:fit_last+1])
    tt=[Yr[i+a.h-1][Mr[i+a.h-1]>0] for i in tr]
else:
    fx=X.reshape(-1,NF); ew=np.log1p(EW); tt=[Yr[i][Mr[i]>0] for i in range(T)]
fmu=fx.mean(0); fsd=fx.std(0)+1e-6; ewmin=ew.min(); ewr=ew.max()-ewmin+1e-6
tt=np.concatenate(tt); tmu=float(tt.mean()); tsd=float(tt.std()+1e-6)
Z=(X-fmu)/fsd
EWn=np.clip((np.log1p(EW)-ewmin)/ewr,0,None)+0.01
rng=np.random.default_rng(a.seed)
E=EI.shape[1]
if a.graph=='real': ei=torch.tensor(np.asarray(EI,dtype=np.int64))
elif a.graph=='random':
    pairs=set(); 
    while len(pairs)<E:
        s_,d_=rng.integers(0,N,2)
        if s_!=d_: pairs.add((int(s_),int(d_)))
    ei=torch.tensor(np.array(sorted(pairs),dtype=np.int64).T)
    EWn=EWn[:,rng.permutation(E)]
elif a.graph=='identity': ei=torch.zeros((2,0),dtype=torch.long)
def mk(i):
    x=np.concatenate([Z[j] for j in range(i-W+1,i+1)],axis=1)
    k=i+a.h-1; m=Mr[k]; y=np.where(m>0,(np.nan_to_num(Yr[k])-tmu)/tsd,0)
    ea=torch.tensor(EWn[i]).unsqueeze(-1) if a.graph!='identity' else torch.zeros((0,1))
    return Data(x=torch.tensor(x),edge_index=ei,edge_attr=ea,y=torch.tensor(np.nan_to_num(y),dtype=torch.float32),mask=torch.tensor(m,dtype=torch.float32),sid=torch.tensor([i]))
D={i:mk(i) for i in np.concatenate([tr,te])}
inC=W*NF; H=a.hidden; dp=a.dropout
class MLP(nn.Module):
    def __init__(s,inc=inC):
        super().__init__(); s.net=nn.Sequential(nn.Linear(inc,H),nn.ReLU(),nn.Dropout(dp),nn.Linear(H,H),nn.ReLU(),nn.Linear(H,1))
    def forward(s,b): return s.net(b.x).squeeze(-1)
class MLPNB(MLP):
    def __init__(s): super().__init__(2*inC)
    def forward(s,b):
        w=b.edge_attr.squeeze(-1); src,dst=b.edge_index
        num=scatter(b.x[src]*w[:,None],dst,0,dim_size=b.x.size(0),reduce='sum'); den=scatter(w,dst,0,dim_size=b.x.size(0),reduce='sum').clamp(min=1e-6)
        return s.net(torch.cat([b.x,num/den[:,None]],1)).squeeze(-1)
class GCN(nn.Module):
    def __init__(s):
        super().__init__(); s.c1=GCNConv(inC,H); s.n1=nn.LayerNorm(H); s.c2=GCNConv(H,H); s.n2=nn.LayerNorm(H); s.out=nn.Linear(H,1); s.d=nn.Dropout(dp)
    def forward(s,b):
        w=b.edge_attr.squeeze(-1) if b.edge_attr.numel() else None
        h=s.d(F.relu(s.n1(s.c1(b.x,b.edge_index,w)))); h=F.relu(s.n2(s.c2(h,b.edge_index,w))); return s.out(h).squeeze(-1)
class SAGE(nn.Module):  # separate self / neighbour weights (GraphConv, mean aggregation, edge-weighted)
    def __init__(s):
        super().__init__(); s.c1=GraphConv(inC,H,aggr='mean'); s.n1=nn.LayerNorm(H); s.c2=GraphConv(H,H,aggr='mean'); s.n2=nn.LayerNorm(H); s.out=nn.Linear(H,1); s.d=nn.Dropout(dp)
    def forward(s,b):
        w=b.edge_attr.squeeze(-1) if b.edge_attr.numel() else None
        h=s.d(F.relu(s.n1(s.c1(b.x,b.edge_index,w)))); h=F.relu(s.n2(s.c2(h,b.edge_index,w))); return s.out(h).squeeze(-1)
class GAT(nn.Module):
    def __init__(s,hc=H//2,heads=4):
        super().__init__(); s.c1=GATConv(inC,hc,heads=heads,dropout=dp,edge_dim=1); s.n1=nn.LayerNorm(hc*heads); s.c2=GATConv(hc*heads,hc,heads=1,dropout=dp,edge_dim=1); s.n2=nn.LayerNorm(hc); s.out=nn.Linear(hc,1)
    def forward(s,b):
        ea=b.edge_attr if b.edge_attr.numel() else None
        h=F.elu(s.n1(s.c1(b.x,b.edge_index,ea))); h=F.elu(s.n2(s.c2(h,b.edge_index,ea))); return s.out(h).squeeze(-1)
M={'mlp':MLP,'mlpnb':MLPNB,'gcn':GCN,'sage':SAGE,'gat':GAT}[a.model]()
lr=a.lr or (0.001 if a.model=='gat' else 0.005)
nparam=sum(p.numel() for p in M.parameters())
g=torch.Generator(); g.manual_seed(a.seed)
L=lambda ids,sh: DataLoader([D[i] for i in ids],batch_size=8,shuffle=sh,generator=g if sh else None)
trl,val,tel=L(itr,True),L(va,False),L(te,False)
opt=torch.optim.Adam(M.parameters(),lr=lr,weight_decay=5e-4)
def ev(loader,ret=False):
    M.eval(); e=0; c=0; P=[]
    with torch.no_grad():
        for b in loader:
            pr=M(b)*tsd+tmu; yt=b.y*tsd+tmu
            e+=((pr-yt).abs()*b.mask).sum().item(); c+=b.mask.sum().item()
            if ret: P.append((b.sid.numpy(),pr.view(-1,N).numpy(),yt.view(-1,N).numpy(),b.mask.view(-1,N).numpy()))
    return (e/c,P) if ret else e/c
best=(1e9,None,0); bad=0; t0=time.time()
for ep in range(1,a.epochs+1):
    M.train()
    for b in trl:
        opt.zero_grad(); pr=M(b); loss=(((pr-b.y)**2)*b.mask).sum()/b.mask.sum().clamp(min=1)
        loss.backward(); torch.nn.utils.clip_grad_norm_(M.parameters(),1.0); opt.step()
    vm=ev(val)
    if vm<best[0]: best=(vm,{k:v.clone() for k,v in M.state_dict().items()},ep); bad=0
    else:
        bad+=1
        if bad>=a.patience: break
M.load_state_dict(best[1]); tm,P=ev(tel,True)
sid=np.concatenate([x[0] for x in P]); pr=np.concatenate([x[1] for x in P]); yt=np.concatenate([x[2] for x in P]); mk_=np.concatenate([x[3] for x in P])
key=f"{a.model}_{a.graph}_{a.norm}_h{a.h}_{a.split}_lag{a.lag}_H{a.hidden}_s{a.seed}{a.tag}"
os.makedirs(a.preds,exist_ok=True)
np.savez_compressed(os.path.join(a.preds,f'{key}.npz'),sid=sid,pr=pr,yt=yt,mk=mk_)
r=dict(key=key,**vars(a),lr_used=lr,test_mae=tm,val_mae=best[0],best_epoch=best[2],epochs_run=ep,params=nparam,secs=round(time.time()-t0))
open(a.out,'a').write(json.dumps(r)+'\n'); print(json.dumps(r))
