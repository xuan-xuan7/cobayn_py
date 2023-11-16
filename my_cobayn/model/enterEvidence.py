import numpy as np
from pgmpy.inference import VariableElimination

def enterEvidence(model, metrics):
    # Normalize metrics
    cleanMetrics = (metrics - model['norm']['mean']) / model['norm']['std']
    cleanMetrics = cleanMetrics[:, model['norm']['cleanMask']]  # cleanMetrics here must be a row vector.

    # Get principal components
    pc = np.dot(cleanMetrics, model['pca']['coeff'])  # pc here must be a row vector.

    evidence = [None] * model['BN']['dag'].shape[0]

    for i in range(model['BN']['dag'].shape[0]):
        if i < model['pca']['length']:
            evidence[i] = pc[i]
        else:
            evidence[i] = None

    engine = VariableElimination(model['BN']['bnet'])
    engine.evidence = evidence

    return engine

# Example usage:
# model = ...  # Obtain the model
# metrics = ...  # Obtain the metrics
# engine = enterEvidence(model, metrics)
# # Use the engine for further inference
