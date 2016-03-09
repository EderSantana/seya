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
        return -K.mean(K.exp(-K.sqr(y_true - y_pred)/sigma), -1)
    return func


def _get_kernel(X, Z, ksize):
    G = K.sum((K.expand_dims(X, dim=1) -
               K.expand_dims(Z, dim=0))**2, axis=-1)  # Gram matrix
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


def gdl(img_shape, alpha=2):
    """Image gradient difference loss

    img_shape: (channels, rows, cols) shape to resize the input
        vectors, we assume they are input flattened in the spatial dimensions.
    alpha: l_alpha norm

    ref: Deep Multi-scale video prediction beyond mean square error,
         by Mathieu et. al.
    """
    def func(y_true, y_pred):
        y_true = K.batch_flatten(y_true)
        y_pred = K.batch_flatten(y_pred)
        Y_true = K.reshape(y_true, (-1, ) + img_shape)
        Y_pred = K.reshape(y_pred, (-1, ) + img_shape)
        t1 = K.pow(K.abs(Y_true[:, :, 1:, :] - Y_true[:, :, :-1, :]) -
                   K.abs(Y_pred[:, :, 1:, :] - Y_pred[:, :, :-1, :]), alpha)
        t2 = K.pow(K.abs(Y_true[:, :, :, :-1] - Y_true[:, :, :, 1:]) -
                   K.abs(Y_pred[:, :, :, :-1] - Y_pred[:, :, :, 1:]), alpha)
        out = K.mean(K.batch_flatten(t1 + t2), -1)
        return out
    return func


def gdl_video(img_shape, alpha=2):
    """Image gradient difference loss for videos

    img_shape: (time, channels, rows, cols) shape to resize the input
        vectors, we assume they are input flattened in the spatial dimensions.
    alpha: l_alpha norm

    ref: Deep Multi-scale video prediction beyond mean square error,
         by Mathieu et. al.
    """
    def func(y_true, y_pred):
        Y_true = K.reshape(y_true, (-1, ) + img_shape)
        Y_pred = K.reshape(y_pred, (-1, ) + img_shape)
        t1 = K.pow(K.abs(Y_true[:, :, :, 1:, :] - Y_true[:, :, :, :-1, :]) -
                   K.abs(Y_pred[:, :, :, 1:, :] - Y_pred[:, :, :, :-1, :]), alpha)
        t2 = K.pow(K.abs(Y_true[:, :, :, :, :-1] - Y_true[:, :, :, :, 1:]) -
                   K.abs(Y_pred[:, :, :, :, :-1] - Y_pred[:, :, :, :, 1:]), alpha)
        out = K.mean(K.batch_flatten(t1 + t2), -1)
        return out
    return func
