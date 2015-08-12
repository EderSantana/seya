import theano.tensor as T


def gaussianKL(y_true, y_pred):
    dim = y_pred.shape[-1] / 2
    mean = y_pred[:, :dim]
    logsigma = y_pred[:, dim:]
    kl = -.5 - logsigma + .5 * (mean**2 + T.exp(2 * logsigma))
    return kl + 0 * y_true.sum()
