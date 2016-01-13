from keras import backend as K


def sum_mse(y_true, y_pred):
    return K.sqr(y_true - y_pred).sum()


def self_cost(y_true, y_pred):
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
        return K.sum(K.exp(-K.sqr(y_true - y_pred)/sigma))
    return func
