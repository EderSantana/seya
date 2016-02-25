import numpy as np
import keras.backend as K
from seya.layers.imageproc import NormLayer
from seya.utils import batchwise_function


def zca_whitening(inputs, epsilon=0.1):
    sigma = np.dot(inputs.T, inputs)/inputs.shape[0]  # Correlation matrix
    U, S, V = np.linalg.svd(sigma)  # Singular Value Decomposition
    epsilon = 0.1  # Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(S + epsilon))), U.T)  # ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs.T).T, ZCAMatrix  # Data whitening


def contrast_normalization(inputs, batch_size=100):
    norm = NormLayer()
    lcn = K.function([norm.get_input()], norm.get_output())
    return batchwise_function(lcn, inputs, batch_size=batch_size)


def global_normalization(inputs, batch_size=100):
    norm = NormLayer(method="gcn")
    lcn = K.function([norm.get_input()], norm.get_output())
    return batchwise_function(lcn, inputs, batch_size=batch_size)
