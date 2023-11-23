import pickle
import warnings
import os
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import MmhcEstimator
# from utils.learnStructMwst import learn_struct_mwst
# from utils.topologicalSort import topological_sort
# from utils.learnStructK2 import learn_struct_K2
# from utils.scoreDag import score_dags
# from pgmpy.models import BayesianModel
from pgmpy.estimators import K2Score
# from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import HillClimbSearch
# from pgmpy.estimators import StructureScore
# from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import BayesianNetwork



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
    PCAlength = 10
    pca = PCA(n_components=PCAlength)
    pcaData = pca.fit_transform(cleanMetricMatrix)
    pcaCoeff = pca.components_
    # print(pcaCoeff, pcaData)
    trainData = np.concatenate((np.tile(pcaData[:, :PCAlength], (bestPerBenchmark, 1)), flagMatrix), axis=1)

    # Build DAG
    Card = np.ones(PCAlength + flags)
    Card[PCAlength:] = 2
    discrete = PCAlength + np.arange(flags)
    types = ['gaussian'] * PCAlength + ['tabular'] * flags
    trainData = pd.DataFrame.from_records(trainData)
    # trainData = trainData.head(2)
    # print(type(trainData[0]))

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
    # model = hc.estimate(scoring_method=scoring_method)

    # print(type(model))
    bn_model = BayesianNetwork(model.edges())
    # print(bn_model.edges())


    # # 最小生成树，连接所有节点且没有环
    # mwst_mat = learn_struct_mwst(trainData)
    
    # # 确定节点间的处理顺序
    # order = topological_sort(mwst_mat)
    # # with open('data/test_train.pkl', 'wb') as f:
    # #     pickle.dump(trainData, f)
    # # with open('data/test_Card.pkl', 'wb') as f:
    # #     pickle.dump(Card, f)
    # # with open('data/test_order.pkl', 'wb') as f:
    # #     pickle.dump(order, f)
    # # 经过K2可以给每个节点选择最优的父节点
    # dag = learn_struct_K2(trainData, Card, order)

    # score = score_dags(trainData, Card, [dag], types)

    # estimator = BayesianEstimator(bn_model, trainData)
    # bn_model.fit(trainData, estimator=BayesianEstimator)
    # for cpd in bn_model.get_cpds():
    #     print(cpd)

    # Build Bayesian network
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
