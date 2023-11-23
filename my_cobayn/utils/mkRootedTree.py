import numpy as np
import networkx as nx

def mk_rooted_tree(G, root):
    n = len(G)
    T = np.zeros((n, n), dtype=int)

    pred = nx.dfs_predecessors(G, source=root)
    for i in pred:
        if pred[i] is not None:
            T[pred[i], i] = 1
    
    T_full = np.full(T.shape, fill_value=0)
    T_full[T.nonzero()] = 1

    return T_full