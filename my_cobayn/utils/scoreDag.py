import numpy as np
from pgmpy.estimators import K2Score
from pgmpy.models import BayesianModel

def score_dags(data, ns, dags, **kwargs):
    n, ncases = data.shape

    # Set default parameters
    type_ = ['tabular'] * n
    params = [{'prior_type': 'dirichlet', 'dirichlet_weight': 1}] * n
    scoring_fn = 'bayesian'
    discrete = list(range(1, n+1))

    # Parse kwargs
    if 'type' in kwargs:
        type_ = kwargs['type']
    if 'params' in kwargs:
        params = kwargs['params']
    if 'scoring_fn' in kwargs:
        scoring_fn = kwargs['scoring_fn']
    if 'discrete' in kwargs:
        discrete = kwargs['discrete']

    u = np.arange(ncases)  # DWH
    isclamped = False  # DWH
    clamped = np.zeros((n, ncases))

    if 'clamped' in kwargs:
        clamped = kwargs['clamped']
        isclamped = True

    NG = len(dags)
    scores = np.zeros(NG)
    for g in range(NG):
        dag = dags[g]
        for j in range(n):
            if isclamped:
                u = np.where(clamped[j] == 0)[0]
            ps = list(dag.predecessors(j))
            scores[g] += score_family(j, ps, type_[j], scoring_fn, ns, discrete, data[:, u], params[j])

    return scores

def score_family(j, ps, node_type, scoring_fn, ns, discrete, data, params):
    if scoring_fn == 'bayesian':
        model = BayesianModel()
        model.add_edges_from([(p, j) for p in ps])
        k2score = K2Score(data=data, node_types=node_type, state_names=ns, complete_samples_only=True)
        score = k2score.score(model)
    else:
        # Handle other scoring functions if needed
        score = 0

    return score