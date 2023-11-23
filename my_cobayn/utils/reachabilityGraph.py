import numpy as np

def reachability_graph(G):
    n = len(G)
    C = np.zeros((n, n))

    if True:
        M = np.linalg.matrix_power(G, n) - np.eye(n)
        C = (M > 0).astype(int)
    else:
        A = G.copy()
        for i in range(n - 1):
            C = C + A
            A = np.matmul(A, G)
        C = (C > 0).astype(int)

    return C