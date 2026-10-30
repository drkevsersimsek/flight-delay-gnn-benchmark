import numpy as np
def load_graphs(path='graphs_hourly_v2.npz'):
    z=np.load(path); a={k:z[k] for k in z.files}   # read each array once
    ts=a['timeslot'].astype('datetime64[ns]'); ei=a['fixed_edge_index'].astype(np.int64)
    G=[dict(node_feat=a['node_feat'][t],edge_weight=a['edge_weight'][t],target=a['target'][t],
            target_mask=a['target_mask'][t],has_dep=a['has_dep'][t],timeslot=ts[t],edge_index=ei)
       for t in range(len(ts))]
    return dict(graphs=G,n_nodes=int(a['n_nodes']),fixed_edge_index=ei)
