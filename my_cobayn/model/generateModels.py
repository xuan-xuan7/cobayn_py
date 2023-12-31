import os
from dataProcessing.getBestSet import getBestSet
from model.getModel import getModel

import numpy as np
import pickle

# import atexit
# import multiprocessing as mp

# # 注册退出时执行的函数
# @atexit.register
# def cleanup():
#     mp.sem_unlink(mp.get_context().SemLock._semlockname)


# import multiprocessing
# from pgmpy.estimators import HillClimbSearch, BicScore
# from pgmpy.models import BayesianModel

# Load initial data
def load_data_from_file(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data
data = load_data_from_file('data/initialData.pkl')
bench = data['bench']
micaMetrics = data['micaMetrics']

# Set the number of best configurations per benchmark
bestPerBenchmark = 25

# Function to exclude application a from the training set
#def getBestSet(bench, bestPerBenchmark, a=None):
#    if a is None:
#        bestSet = bench[:bestPerBenchmark, :]
#        ExT = np.delete(bench, range(bestPerBenchmark), axis=0)
#    else:
#        bestSet = np.concatenate((bench[:bestPerBenchmark*(a-1), :], bench[bestPerBenchmark*a:, :]), axis=0)
#        ExT = bench[bestPerBenchmark*(a-1):bestPerBenchmark*a, :]
#    return bestSet, ExT

# Generate the BN model
bestSet, _ = getBestSet(bench, bestPerBenchmark)
# print(bestSet)
BNmodel = getModel(bestSet, micaMetrics)
print('Populating BNmodel using selected bestPerApplication is finished!')
print('Now, generating the cross-validation models for each application.')

# Start the parallel session
# try:
#     multiprocessing.set_start_method('spawn')
#     pool = multiprocessing.Pool()
# except:
#     pass

# Generate cross-validation models
def trainBNmodel(a):
    bestSet, _ = getBestSet(bench, bestPerBenchmark, a+1)
    BNmodel_loo = getModel(bestSet, micaMetrics)
    print(f'Application {a+1} is trained!')
    return BNmodel_loo

num_apps = bench.shape[0]
BNmodel_loo = []
for a in range(num_apps):
    BNmodel_loo[a] = map(trainBNmodel, a)

# Save the models
# np.savez('data/models.npz', BNmodel_loo=BNmodel_loo, BNmodel=BNmodel)
if os.path.exists("data/BNmodel.pkl"):
    pass
else:
    with open('data/BNmodel.pkl', 'wb') as f:
        pickle.dump(BNmodel, f)

if os.path.exists("data/BNmodel_loo.pkl"):
    pass
else:
    with open('data/BNmodel_loo.pkl', 'wb') as f:
        pickle.dump(BNmodel_loo, f)

print('BN validated with leave-one-out')