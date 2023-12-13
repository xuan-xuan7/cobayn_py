import pickle
import warnings
import os
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pgmpy.estimators import MmhcEstimator
from pgmpy.estimators import K2Score
from pgmpy.estimators import HillClimbSearch
from pgmpy.models import BayesianNetwork
from sklearn.preprocessing import StandardScaler
from utils.getContinuousCPD import calculate_continuous_cpds
from utils.getDiscreteCPD import calculate_discrete_cpd


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
    pca = PCA()
    pcaData = pca.fit_transform(cleanMetricMatrix)
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    PCAlength = np.argmax(explained_variance_ratio >= 0.95) + 1
    pca = PCA(n_components=PCAlength)
    pcaData = pca.fit_transform(cleanMetricMatrix)
    pcaCoeff = pca.components_

    scaler = StandardScaler()
    normalizedData = scaler.fit_transform(pcaData)

    # print(pcaCoeff, pcaData)
    trainData = np.concatenate((np.tile(normalizedData[:, :PCAlength], (bestPerBenchmark, 1)), flagMatrix), axis=1)

    # np.savetxt("./data/trainData.csv", trainData, delimiter=',', header=header, comments='')
    # trainData = pd.read_csv("./data/trainData.csv")
    # # trainData = trainData.astype(int)
    # print(trainData)
    
    # Card = np.ones(PCAlength + flags)
    # Card[PCAlength:] = 2
    # discrete = PCAlength + np.arange(flags)
    types = [0] * PCAlength + [1] * flags
    # print(types)

    trainData = pd.DataFrame.from_records(trainData)

    # if os.path.exists("data/test_train.pkl"):
    #     with open("data/test_train.pkl", 'rb') as f:
    #         trainData = pickle.load(f)
    # else:
    #     with open("data/test_train.pkl", 'wb') as f:
    #         pickle.dump(trainData, f)

    if os.path.exists("data/skeleton.pkl"):
        with open("data/skeleton.pkl", 'rb') as f:
            skeleton = pickle.load(f)
        # print("Part 1) Skeleton: ", skeleton.edges())
    else:
        mmhc = MmhcEstimator(trainData)
        skeleton = mmhc.mmpc()
        with open('data/skeleton.pkl', 'wb') as f:
            pickle.dump(skeleton, f)

    scoring_method = K2Score(data=trainData)
    hc = HillClimbSearch(data=trainData)
    model = hc.estimate(tabu_length=10, white_list=skeleton.to_directed().edges(), scoring_method=scoring_method)

    bn_model = BayesianNetwork(model.edges())
    
    # Learn parameters
    # estimator = MaximumLikelihoodEstimator(bn_model, trainData)

    cpds = {}
    for i in range(PCAlength + flags):
        if i < PCAlength:
            continue
            cpds[i] = calculate_continuous_cpds(trainData, bn_model, types, i)
        else:
            cpds[i] = calculate_discrete_cpd(trainData, bn_model, types, i)
    # cpds = calculate_continuous_cpds(trainData, bn_model, types, 0)


    # Return the model
    # model = {}
    # model['norm'] = {'mean': meanMetric, 'std': stdMetric, 'cleanMask': cleanMask}
    # model['pca'] = {'length': PCAlength, 'coeff': pcaCoeff, 'pcaData': pcaData}
    # model['trainData'] = {'trainData': trainData, 'bestSet': bestSet, 'metrics': metrics}
    # model['BN'] = {'dag': dag, 'score': score, 'bnet': bnet}
    # print(cpds)

    return bn_model

# Example usage:
# bestSet = ...  # Obtain the bestSet matrix
# metrics = ...  # Specify the number of metrics
# micaBNmodel = getModel(bestSet, metrics)
