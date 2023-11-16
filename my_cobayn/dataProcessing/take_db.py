import numpy as np
import pandas as pd
import pickle

# DSE exploration file
E = pd.read_csv('./data/cBench_onPandaboard_24app_5ds.csv', delimiter=',')  # execution time data collection file
# 一个测试项测试了几次
datasets = len(E.select_dtypes(include='number').columns) - 1  # last column is code size
compilerFlags = E.shape[1] - datasets - 2  # last column is code size & first is app name
# 几个测试项
applications = len(np.unique(E.iloc[:, 0]))
# 每个测试项测试了多少种情况
experiments = int(E.shape[0] / applications)

appNames_flags = E.iloc[:, 0].values
txtFlags = E.iloc[:, 1:(compilerFlags + 1)].values
flagNames = E.columns.tolist()[1:(compilerFlags + 1)]
flags = ~(txtFlags == 'X')
exTimes = E.iloc[:, (compilerFlags + 1): -1].values


M = pd.read_csv('./data/ft_Milepost_cbench.csv', delimiter=',')  # software characterization file

if not (M.shape[0] == datasets * applications):
    raise ValueError('The app characterization database is not compliant with the exploration database')

# 特征数量
micaMetrics = M.shape[1] - 2  # first two are appname and dataset.
header = M.columns
# 特征名字
micaNames = list(header[2:(micaMetrics + 2)])
# 特征数据矩阵
micas = M.iloc[:, 2:(micaMetrics + 2)].values

# Choose one from the 5 different methods Deafult is 1.

# 1
# Normalizing by the first column
micasNorm = micas / np.tile(micas[:, 0][:, np.newaxis], (1, micaMetrics))

# 2
# When we have them both as stat/dynamic-analyzer

# ONLY FOR cbench both
# micasNorm = np.concatenate((micas[:, :45] / np.tile(micas[:, 0], (1, 45)), micas[:, 46:62] / np.tile(micas[:, 46], (1, 17))), axis=1)

# ONLY FOR poly both
# micasNorm = np.concatenate((micas[:, :16] / np.tile(micas[:, 0], (1, 16)), micas[:, 17:31] / np.tile(micas[:, 17], (1, 15))), axis=1)

# 3
# matlab normc function
# micasNorm = micas / np.linalg.norm(micas, axis=0)

# 4
# normalizing the micas using (Xi - mean(column of Xi)) / std(all_data)
# micasNorm = (micas - np.mean(micas, axis=0)) / np.std(micas)

# 5
# normalizing the micas using (Xi - mean(column of Xi)) / std(columns)
# micasNorm = (micas - np.mean(micas, axis=0)) / np.std(micas, axis=0)

i_m = 1
i_e = 1
bench = []
for a in range(applications):
    bench.append([])
    appName = M.iloc[i_m, 0]
    if not appName == appNames_flags[i_e - 1]:
        raise ValueError('appName differs from experiments to mica')

    for d in range(datasets):
        bench[a].append({
            'application': appName,
            'dataset': d + 1,
            'mica': micas[i_m - 1, :],
            'micaNorm': micasNorm[i_m - 1, :],
            'DB': flags[i_e - 1:i_e + experiments - 1, :],
            'Y': exTimes[i_e - 1:i_e + experiments - 1, d]
        })
        i_m += 1
    i_e += experiments
    
data = {'bench': bench, 'micaMetrics': micaMetrics, 'experiments': experiments, 'datasets': datasets,
        'applications': applications}

with open('data/initialData.pkl', 'wb') as f:
    pickle.dump(data, f)

