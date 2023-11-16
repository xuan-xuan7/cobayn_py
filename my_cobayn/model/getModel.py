import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
from pgmpy.estimators import K2Score
from pgmpy.estimators import MaximumLikelihoodEstimator
from utils.learnStructMwst import learn_struct_mwst
# from pgmpy.models import BayesianModel
# from pgmpy.estimators import K2Score
# from pgmpy.estimators import BayesianEstimator
# from pgmpy.estimators import PC, HillClimbSearch
# from pgmpy.estimators import StructureScore
# from pgmpy.sampling import BayesianModelSampling

def getModel(bestSet, metrics):
    bestPerBenchmark = 25

    metricMatrix = bestSet[0::bestPerBenchmark, 0:metrics]
    flagMatrix = bestSet[:, metrics:]
    flags = len(flagMatrix[0])

    # Standardize metrics
    meanMetric = np.mean(metricMatrix, axis=0)
    stdMetric = np.std(metricMatrix, axis=0)
    cleanMetricMatrix = (metricMatrix - np.tile(meanMetric, (metricMatrix.shape[0], 1))) / np.tile(stdMetric, (metricMatrix.shape[0], 1))
    cleanMask = ~np.isnan(cleanMetricMatrix).any(axis=0)
    cleanMetricMatrix = cleanMetricMatrix[:, cleanMask]

    # Apply PCA
    PCAlength = 5
    pca = PCA(n_components=PCAlength)
    pcaData = pca.fit_transform(cleanMetricMatrix)
    pcaCoeff = pca.components_
    # print(pcaCoeff, pcaData)
    trainData = np.concatenate((np.tile(pcaData[:, :PCAlength], (bestPerBenchmark, 1)), flagMatrix + 1), axis=1)

    # Build DAG
    Card = np.ones(PCAlength + flags)
    Card[PCAlength:] = 2
    discrete = PCAlength + np.arange(flags)
    types = ['gaussian'] * PCAlength + ['tabular'] * flags

    trainData = pd.DataFrame(trainData)

    k2_score = K2Score(trainData)

    mwst_model = learn_struct_mwst(trainData)
    order = list(nx.topological_sort(mwst_model))
    dag = learn_struct_K2(trainData, Card, order, types, k2_score)
    score = score_dags(trainData, Card, [dag], types, bic_score)

    # Build Bayesian network
    bnet = mk_bnet(dag, Card, types)
    for i in range(PCAlength + flags):
        if i < PCAlength:
            bnet.CPD[i] = GaussianCPD(bnet, i)
        elif not np.isnan(parents(dag, i)):
            bnet.CPD[i] = SoftmaxCPD(bnet, i)
        else:
            bnet.CPD[i] = TabularCPD(bnet, i)
    
    # Learn parameters
    mle = MaximumLikelihoodEstimator(bnet, trainData)
    bnet = mle.estimate_parameters(trainData)

    # Return the model
    model = {}
    model['norm'] = {'mean': meanMetric, 'std': stdMetric, 'cleanMask': cleanMask}
    model['pca'] = {'length': PCAlength, 'coeff': pcaCoeff, 'pcaData': pcaData}
    model['trainData'] = {'trainData': trainData, 'bestSet': bestSet, 'metrics': metrics}
    model['BN'] = {'dag': dag, 'score': score, 'bnet': bnet}

    return model

# Example usage:
# bestSet = ...  # Obtain the bestSet matrix
# metrics = ...  # Specify the number of metrics
# micaBNmodel = getModel(bestSet, metrics)
