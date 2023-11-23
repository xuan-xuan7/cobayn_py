from utils.reachabilityGraph import reachability_graph
from utils.dfs import dfs

import numpy as np

def acyclic(adj_mat, directed=True):
    adj_mat = np.array(adj_mat, dtype=float)
    if not directed:
        d, pre, post, cycle = dfs(adj_mat, directed=directed)
        b = not cycle
    else:
        R = reachability_graph(adj_mat)
        b = not np.any(np.diag(R) == 1)
    
    return b