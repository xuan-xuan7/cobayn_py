import numpy as np
import networkx as nx
from pgmpy.estimators import BicScore


def learn_struct_mwst(data):

    G = nx.Graph()
    N = data.shape[1]

    bic_score = BicScore(data)
    score_mat = np.zeros((N, N))

    for i in range(N-1):
        score2 = bic_score.local_score(data.columns[i], [])
        for j in range(i+1, N):
            score1 = bic_score.local_score(data.columns[i], [data.columns[j]])
            score = score2 - score1
            score_mat[i, j] = score
            score_mat[j, i] = score

    for i in range(N-1):
        for j in range(i+1, N):
            score = score_mat[i, j]
            G.add_edge(i, j, weight=score)

    model = nx.minimum_spanning_tree(G)

    return model