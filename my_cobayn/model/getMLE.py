import numpy as np
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel

def getMLE(model, metrics):
    # Enter evidence
    engine = enterEvidence(model, metrics)

    # Normalize metrics
    cleanMetrics = (metrics - model['norm']['mean']) / model['norm']['std']
    cleanMetrics = cleanMetrics[:, model['norm']['cleanMask']]

    # Get principal components
    pc = np.dot(cleanMetrics, model['pca']['coeff'])

    evidence = [None] * model['BN']['dag'].shape[0]

    for i in range(model['BN']['dag'].shape[0]):
        if i < model['pca']['length']:
            evidence[i] = pc[i]
        else:
            evidence[i] = None

    # Find the most probable explanation
    MLE = engine.map_query(variables=[], evidence=evidence)

    return MLE

# Example usage:
# model = ...  # Obtain the model
# metrics = ...  # Obtain the metrics
# MLE = getMLE(model, metrics)
