import theano.tensor as T


def gaussianKL(dumb, y_pred):
    # Assumes: y_pred = T.concatenate([mean, logsigma], axis=-1)
    dim = y_pred.shape[1] / 2
    mean = y_pred[:, :dim]
    logsigma = y_pred[:, dim:]
    # See Variational Auto-Encoding Bayes by Kingma and Welling.
    kl = -.5 - logsigma + .5 * (mean**2 + T.exp(2 * logsigma))
    return kl.mean(axis=-1) + 0 * dumb.sum()
