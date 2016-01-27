import numpy as np

from keras import backend as K


def sum_mse(y_true, y_pred):
    return K.sqr(y_true - y_pred).sum()


def self_cost(y_true, y_pred):
    # output itself is cost and must be minimized
    return K.sum(y_pred) + K.sum(y_true)*0


def gaussianKL(dumb, y_pred):
    # Assumes: y_pred = T.concatenate([mean, logsigma], axis=-1)
    dim = y_pred.shape[1] / 2
    mean = y_pred[:, :dim]
    logsigma = y_pred[:, dim:]
    # See Variational Auto-Encoding Bayes by Kingma and Welling.
    kl = -.5 - logsigma + .5 * (mean**2 + K.exp(2 * logsigma))
    return K.mean(kl, axis=-1) + 0 * K.sum(dumb)


def correntropy(sigma=1.):
    def func(y_true, y_pred):
        return -K.sum(K.exp(-K.sqr(y_true - y_pred)/sigma))
    return func


def _get_kernel(X, Z, ksize):
    G = K.sum((K.expand_dims(X, dim=1) - K.expand_dims(Z, dim=0))**2, axis=-1)  # Gram matrix
    G = K.exp(-G/(ksize)) / K.sqrt(2*np.pi*ksize)
    return G


def ITLeuclidean(ksize=1.):
    def func(y_true, y_pred):
        Gxx = _get_kernel(y_true, y_true, ksize)
        Gzz = _get_kernel(y_pred, y_pred, ksize)
        Gxz = _get_kernel(y_true, y_pred, ksize)
        cost = K.mean(Gxx) + K.mean(Gzz) - 2*K.mean(Gxz)
        return cost
    return func


def ITLcsd(ksize=1.):
    def func(y_true, y_pred):
        Gxx = _get_kernel(y_true, y_true)
        Gzz = _get_kernel(y_pred, y_pred)
        Gxz = _get_kernel(y_true, y_pred)
        cost = K.log(K.sqrt(K.mean(Gxx)*K.mean(Gzz)) /
                     K.mean(Gxz))
        return cost
    return func
