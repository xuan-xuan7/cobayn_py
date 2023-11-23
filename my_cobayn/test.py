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

import pandas as pd
import numpy as np
import pickle
from pgmpy.estimators import K2Score, BicScore
from pgmpy.estimators import MmhcEstimator
from pgmpy.estimators import HillClimbSearch


def structuralLearning_Hybrid():
    """
    MMHC算法[3]结合了基于约束和基于分数的方法。它有两部分:
        1. 使用基于约束的构造过程MMPC学习无向图骨架
        2. 基于分数的优化(BDeu分数+修改爬山)
    """
    print("===================混合方法=================================")
    # 实验数据生成
    data = pd.DataFrame(np.random.rand(25, 8), columns=list('ABCDEFGH'))
    data['A'] += data['B'] + data['C']
    data['H'] = data['G'] - data['A']
    data['E'] *= data['F']
    test = np.random.rand(25, 8)
    print(type(test[0]))
    # print(type(data[0]))
    # with open("data/test_train.pkl", 'rb') as f:
    #         data = pickle.load(f)
    # with open("data/skeleton.pkl", 'rb') as f:
    #         skeleton = pickle.load(f)
    # print(data.columns.values)
    # print(data)
    # 构建无向图骨架
    mmhc = MmhcEstimator(data)
    skeleton = mmhc.mmpc()
    print("Part 1) Skeleton: ", skeleton.edges())
    # 基于分数优化
    # use hill climb search to orient the edges:
    hc = HillClimbSearch(data)
    model = hc.estimate(tabu_length=10, white_list=skeleton.to_directed().edges(), scoring_method=K2Score(data))
    print("Part 2) Model:    ", model.edges())

if __name__ == "__main__":
    # parameterLearning()
    # structuralLearning_Score()
    # structuralLearning_Constraint()
    structuralLearning_Hybrid()