import numpy as np
import scipy.io as sio

iteration = []

# Generates the complete prediction vector of (2^7 = 128)
# a bit slow but need to do it once and use it later
# on the comparison. i.e.: Top-N will be the first N rows
samples = 128

# Generates the predictions
predTable = useModels(samples, bench, BNmodel_loo)

# Sort out the prediction table into the different applications and save
# them to COBAYN structure
COBAYN = np.empty((len(bench), len(bench[0]), samples, predTable.shape[1]))
pp = 0
for a in range(len(bench)):
    for d in range(len(bench[0])):
        for s in range(samples):
            COBAYN[a, d, s, :] = predTable[pp, :]
            pp += 1

# Finally, save the predictions
data = {'COBAYN': COBAYN}
sio.savemat('data/COBAYN_predictions.mat', data)
