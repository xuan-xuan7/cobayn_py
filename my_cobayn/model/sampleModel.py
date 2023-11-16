import numpy as np
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import TabularCPD

def sampleModel(model, metrics, nsamples):
    # Get the MLE point
    sam_ = getMLE(model, metrics)
    sample = np.array([cat(sam_) for sam_ in sam_])

    # Normalize metrics
    cleanMetrics = (metrics - model['norm']['mean']) / model['norm']['std']
    cleanMetrics = cleanMetrics[:, model['norm']['cleanMask']]

    # Get principal components
    pc = cleanMetrics.dot(model['pca']['coeff'])

    evidence = [pc[i] if i < model['pca']['length'] else [] for i in range(len(model['BN']['dag']))]

    s = 1
    while s < nsamples:
        csample = sample_bnet(model['BN']['bnet'], evidence=evidence)
        sam = np.array([cat(csample_) for csample_ in csample])

        if not np.any(np.all(sam == sample, axis=1)):
            sample = np.vstack((sample, sam))
            s += 1

    return sample

# Helper function to convert cell array to numpy array
def cat(cell_array):
    return np.concatenate(cell_array, axis=0)

# Example usage:
# model = ...  # The Bayesian network model
# metrics = ...  # The metrics data
# nsamples = ...  # Number of samples to generate
# samples = sampleModel(model, metrics, nsamples)
