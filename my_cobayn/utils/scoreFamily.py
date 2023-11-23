import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BicScore, BdeuScore, K2Score
from pgmpy.factors.continuous import ContinuousFactor
from utils.mkBnet import mk_bnet

def gaussian_CPD(bnet, node):
    # 获取节点的状态空间大小
    node_size = bnet.node_sizes[node]
    
    # 创建节点的条件线性高斯分布的CPD
    cpd = ContinuousFactor([node], [], [node_size])
    
    # 随机生成参数
    mean = np.random.randn(node_size)
    covariance = np.eye(node_size)  # 假设协方差矩阵为单位矩阵
    
    # 将参数设置到CPD中
    cpd.set_parameters([mean, covariance])
    
    return cpd


def score_family(j, ps, node_type, scoring_fn, ns, discrete, data):
    n, ncases = data.shape
    dag = np.zeros((n, n))
    if len(ps) > 0:
        dag[ps, j] = 1
        ps = sorted(ps)

    bnet = mk_bnet(dag, ns, discrete)

    bnet.add_cpds(cpd)

    if node_type == 'tabular':
        from pgmpy.estimators import MaximumLikelihoodEstimator
        cpd = MaximumLikelihoodEstimator(bnet, data).estimate_cpd(j, prior_type='dirichlet', pseudo_counts=1)
        bnet.add_cpds(cpd)
    else:
        raise ValueError("Unsupported node type: {}".format(node_type))

    score = None
    if scoring_fn == 'bic':
        score = BicScore(data).score(bnet)
    elif scoring_fn == 'bayesian':
        score = BdeuScore(data).score(bnet)
    else:
        raise ValueError("Unrecognized scoring function: {}".format(scoring_fn))

    return score