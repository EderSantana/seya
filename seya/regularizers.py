import numpy as np

from keras import backend as K
from keras.regularizers import Regularizer


class GaussianKL(Regularizer):
    """ KL-divergence between two gaussians.
    Useful for Variational AutoEncoders.
    Use this as an activation regularizer

    Parameters:
    -----------
    mean, logsigma: parameters of the input distributions
    prior_mean, prior_logsigma: paramaters of the desired distribution (note the
    log on logsigma)
    regularizer_scale: Rescales the regularization cost. Keep this 1 for most cases.

    Notes:
    ------
    See seya.layers.variational.VariationalDense for usage example

    """
    def __init__(self, mean, logsigma, prior_mean=0, prior_logsigma=0,
                 regularizer_scale=1):
        self.regularizer_scale = regularizer_scale
        self.mean = mean
        self.logsigma = logsigma
        self.prior_mean = prior_mean
        self.prior_logsigma = prior_logsigma
        super(GaussianKL, self).__init__()

    def __call__(self, loss):
        # See Variational Auto-Encoding Bayes by Kingma and Welling.
        mean, logsigma = self.mean, self.logsigma
        kl = (self.prior_logsigma - logsigma +
              0.5 * (-1 + K.exp(2 * logsigma) + (mean - self.prior_mean) ** 2) /
              K.exp(2 * self.prior_logsigma))
        loss += K.mean(kl) * self.regularizer_scale
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__}


class ExponentialKL(Regularizer):
    """ KL-divergence between two exponentially distrubted random variables.
    Useful for Variational AutoEncoders.
    Use this as an activation regularizer

    Parameters:
    -----------
    _lambda: parameter of the input distributions
    prior_lambda: paramater of the desired distribution (scale or rate)
    regularizer_scale: Rescales the regularization cost. Keep this 1 for most cases.

    Notes:
    ------
    See seya.layers.variational.VariationalExp for usage example

    """
    def __init__(self, _lambda, prior_lambda=1.,
                 regularizer_scale=1):
        self.regularizer_scale = regularizer_scale
        self._lambda = _lambda
        self.prior_lambda = prior_lambda
        super(ExponentialKL, self).__init__()

    def __call__(self, loss):
        # See Variational Auto-Encoding Bayes by Kingma and Welling.
        kl = (K.log(self._lambda) - K.log(self.prior_lambda) +
              self.prior_lambda/self._lambda - 1)
        loss += K.mean(kl) * self.regularizer_scale
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__}


class LambdaRegularizer(Regularizer):
    def __init__(self, cost):
        super(LambdaRegularizer, self).__init__()
        self.cost = cost

    def __call__(self, loss):
        loss += self.cost
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__}


class CorrentropyActivityRegularizer(Regularizer):
    def __init__(self, scale, sigma=1.):
        super(CorrentropyActivityRegularizer, self).__init__()
        self.sigma = sigma
        self.scale = scale

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        output = self.layer.get_output(True)
        loss += self.scale * correntropy(output, self.sigma)
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "sigma": self.sigma,
                "scale": self.scale}


class CorrentropyWeightRegularizer(Regularizer):
    def __init__(self, scale, sigma=1):
        super(CorrentropyWeightRegularizer, self).__init__()
        self.sigma = sigma
        self.scale = scale

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        loss += self.scale * correntropy(self.p, self.sigma)
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "sigma": self.sigma,
                "scale": self.scale}


def correntropy(x, sigma):
    return -K.sum(K.mean(K.exp(x**2/sigma), axis=0)) / K.sqrt(2*np.pi*sigma)
