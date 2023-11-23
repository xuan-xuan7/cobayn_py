from utils.topologicalSort import topological_sort
from utils.acyclic import acyclic

def mk_bnet(dag, node_sizes, discrete):
    n = len(dag)

    # Default values for parameters
    bnet = {
        'equiv_class': list(range(n)),
        'dnodes': list(range(n)),
        'observed': [],
        'names': []
    }

    bnet['dnodes'] = discrete

    bnet['observed'] = sorted(bnet['observed'])
    bnet['hidden'] = list(set(range(n)).difference(bnet['observed']))
    bnet['hidden_bitv'] = [1 if i in bnet['hidden'] else 0 for i in range(n)]
    bnet['dag'] = dag
    bnet['node_sizes'] = list(node_sizes)

    bnet['cnodes'] = list(set(range(n)).difference(bnet['dnodes']))

    bnet['parents'] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if dag[i][j] == 1:
                bnet['parents'][j].append(i)

    E = max(bnet['equiv_class'])
    mem = [[] for _ in range(E)]
    for i in range(n):
        e = bnet['equiv_class'][i]
        mem[e].append(i)
    bnet['members_of_equiv_class'] = mem

    bnet['CPD'] = [None] * E

    bnet['rep_of_eclass'] = [None] * E
    for e in range(E):
        mems = bnet['members_of_equiv_class'][e]
        bnet['rep_of_eclass'][e] = mems[0]

    # Check acyclicity of the graph
    directed = True
    if not acyclic(dag, directed):
        raise ValueError('Graph must be acyclic')

    bnet['order'] = topological_sort(dag)

    return bnet