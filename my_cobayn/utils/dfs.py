import networkx as nx

def dfs(adj_mat, start=None, directed=True):
    G = nx.DiGraph(adj_mat) if directed else nx.Graph(adj_mat)
    d = {}
    pre = []
    post = []
    cycle = False
    f = {}
    pred = {}

    def dfs_visit(u):
        nonlocal d, pre, post, cycle, f, pred

        pre.append(u)
        d[u] = len(pre)

        for v in G.neighbors(u):
            if v not in d:  # not visited v before (tree edge)
                pred[v] = u
                dfs_visit(v)
            elif v in pre:  # back edge - v has been visited, but is still open
                cycle = True

        post.append(u)
        f[u] = len(post)

    if start is not None:
        dfs_visit(start)

    for u in G.nodes():
        if u not in d:
            dfs_visit(u)

    return d, pre, post, cycle, f, pred