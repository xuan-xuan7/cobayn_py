def topological_sort(adj_mat):
    
    n = len(adj_mat)
    indeg = [0] * n
    zero_indeg = []  # a stack of nodes with no parents

    for i in range(n):
        indeg[i] = sum(adj_mat[j][i] for j in range(n))
        if indeg[i] == 0:
            zero_indeg.append(i)

    t = 0
    order = [0] * n

    while zero_indeg:
        v = zero_indeg.pop(0)
        order[t] = v
        t += 1

        for c in range(n):
            if adj_mat[v][c] == 1:
                indeg[c] -= 1
                if indeg[c] == 0:
                    zero_indeg.insert(0, c)

    return order