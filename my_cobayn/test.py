# import pickle

# import initMatEnv
# print('(PHASE-1) All Paths set')

# def load_data_from_file(filename):
#     with open(filename, 'rb') as file:
#         data = pickle.load(file)
#     return data
# trainData = load_data_from_file('data/test_train.pkl')
# Card = load_data_from_file('data/test_card.pkl')
# order = load_data_from_file('data/test_order.pkl')

# print(trainData, Card, order)

# import pandas as pd
# import numpy as np
# import pickle
# from pgmpy.estimators import K2Score, BicScore
# from pgmpy.estimators import MmhcEstimator
# from pgmpy.estimators import HillClimbSearch


# def structuralLearning_Hybrid():
#     """
#     MMHC算法[3]结合了基于约束和基于分数的方法。它有两部分:
#         1. 使用基于约束的构造过程MMPC学习无向图骨架
#         2. 基于分数的优化(BDeu分数+修改爬山)
#     """
#     print("===================混合方法=================================")
#     # 实验数据生成
#     data = pd.DataFrame(np.random.rand(25, 8), columns=list('ABCDEFGH'))
#     data['A'] += data['B'] + data['C']
#     data['H'] = data['G'] - data['A']
#     data['E'] *= data['F']
#     test = np.random.rand(25, 8)
#     print(type(test[0]))
#     # print(type(data[0]))
#     # with open("data/test_train.pkl", 'rb') as f:
#     #         data = pickle.load(f)
#     # with open("data/skeleton.pkl", 'rb') as f:
#     #         skeleton = pickle.load(f)
#     # print(data.columns.values)
#     # print(data)
#     # 构建无向图骨架
#     mmhc = MmhcEstimator(data)
#     skeleton = mmhc.mmpc()
#     print("Part 1) Skeleton: ", skeleton.edges())
#     # 基于分数优化
#     # use hill climb search to orient the edges:
#     hc = HillClimbSearch(data)
#     model = hc.estimate(tabu_length=10, white_list=skeleton.to_directed().edges(), scoring_method=K2Score(data))
#     print("Part 2) Model:    ", model.edges())

# if __name__ == "__main__":
#     # parameterLearning()
#     # structuralLearning_Score()
#     # structuralLearning_Constraint()
#     structuralLearning_Hybrid()

# import pickle
# import numpy as np
# import pandas as pd
# from sklearn.decomposition import PCA
# from pgmpy.estimators import MaximumLikelihoodEstimator
# from pgmpy.estimators import MmhcEstimator
# from pgmpy.estimators import K2Score
# from pgmpy.estimators import HillClimbSearch
# from pgmpy.models import BayesianNetwork
# from pgmpy.estimators import BayesianEstimator
# from pgmpy.factors.continuous import ContinuousFactor


# with open("data/skeleton.pkl", 'rb') as f:
#     skeleton = pickle.load(f)

# scoring_method = K2Score(data=trainData)
# hc = HillClimbSearch(data=trainData)
# model = hc.estimate(tabu_length=10, white_list=skeleton.to_directed().edges(), scoring_method=scoring_method)
# bn_model = BayesianNetwork(model.edges())

# import numpy as np
# from scipy.stats import multivariate_normal, multinomial, bernoulli

# # 假设数据集为一个二维数组，每一列是一个节点的数据
# dataset = np.array([
#     [1.2, 0, 1, 1],
#     [0.8, 1, 1, 0],
#     [2.1, 1, 0, 1]
# ])

# # 定义节点之间的有向无环图关系
# graph = {
#     0: [],       # 节点0没有父节点
#     1: [],       # 节点1没有父节点
#     2: [0, 1],   # 节点2的父节点是节点0和节点1
#     3: [2]       # 节点3的父节点是节点2
# }

# # 假设节点0和节点1是高斯分布节点，节点2和节点3是离散节点
# gaussian_nodes = [0, 1]  # 高斯分布节点的索引
# discrete_nodes = [2, 3]  # 离散节点的索引

# # 构建高斯分布节点的CPD
# gaussian_cpds = []

# for node in gaussian_nodes:
#     parents = graph[node]
#     if len(parents) == 0:  # 没有父节点，直接使用高斯分布参数
#         data = dataset[:, node]
#         mean = np.mean(data)
#         cov = np.cov(data, rowvar=False)
#         cpd = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
#     else:  # 有父节点，构建多元高斯分布
#         parent_data = dataset[:, parents]
#         data = np.concatenate((parent_data, dataset[:, node].reshape(-1, 1)), axis=1)
#         mean = np.mean(data, axis=0)
#         cov = np.cov(data, rowvar=False)
#         cpd = multivariate_normal(mean=mean, cov=cov, allow_singular=True)

#     # 打印高斯分布节点的CPD
#     print(f"Gaussian Node {node+1} CPD:")
#     print(cpd)

#     gaussian_cpds.append(cpd)

# # 构建离散节点的CPD
# discrete_cpds = []

# for node in discrete_nodes:
#     parents = graph[node]
#     parent_data = dataset[:, parents]
#     data = np.concatenate((parent_data, dataset[:, node].reshape(-1, 1)), axis=1)
#     unique_values, counts = np.unique(data, axis=0, return_counts=True)

#     if len(unique_values) > 2:  # 多值离散节点
#         probabilities = counts / np.sum(counts)
#         cpd = multinomial(n=1, p=probabilities)

#         # 打印多值离散节点的CPD
#         print(f"Multinomial Discrete Node {node+1} CPD:")
#         print(cpd.p)

#     else:  # 二值离散节点
#         p_success = counts[1] / np.sum(counts)
#         cpd = bernoulli(p=p_success)

#         # 打印二值离散节点的CPD
#         print(f"Bernoulli Discrete Node {node+1} CPD:")
#         print(f"P(X = 1) = {cpd.p}")

#     discrete_cpds.append(cpd)


import numpy as np
from scipy.stats import norm

# 定义父节点的取值
parent_values = [0, 1, 2]

# 为每个父节点取值组合计算连续节点的参数
mean_params = [1.0, 2.0, 3.0]
std_params = [0.5, 0.8, 1.2]

def calculate_cpd(parent_value, node_value):
    # 获取父节点取值对应的参数
    mean = mean_params[parent_value]
    print(mean)
    std = std_params[parent_value]
    print(std)
  
    # 计算连续节点在给定父节点取值下的条件概率分布
    probability = norm.pdf(node_value, mean, std)
  
    return probability

# 假设连续节点的取值为2.5，父节点的取值为1
node_value = 2.5
parent_value = 1

# 计算连续节点在给定父节点取值下的条件概率
cpd = calculate_cpd(parent_value, node_value)

print(cpd)