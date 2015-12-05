from keras import backend as K
from keras.regularizers import Regularizer


class GaussianKL(Regularizer):
    """ KL-divergence between two gaussians.
    Useful for Variational AutoEncoders.
    Use this as an activation regularizer

    Parameters:
    -----------
    mean, logsigma: parameters of the input distributions
    prior_mean, prior_logsigma: paramaters of the desired distribution
    regularizer_scale: Rescales the regularization cost. Keep this 1 for most cases.

    Notes:
    ------
    See seya.layers.variational.VariationalDense for usage example

    """
    def __init__(self, mean, logsigma, prior_mean=0, prior_logsigma=1,
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
        kl = (self.prior_logsigma - logsigma
              + 0.5 * (K.exp(2 * logsigma) + (mean - self.prior_mean) ** 2)
              / K.exp(2 * self.prior_logsigma))
        loss += kl.mean() * self.regularizer_scale
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
