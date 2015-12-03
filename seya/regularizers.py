from keras import backend as K
from keras.regularizers import Regularizer


class GaussianKL(Regularizer):
    """ KL-divergence between two gaussians.
    Useful for Variational AutoEncoders.
    Use this as an activation regularizer
    """
    def __init__(self, mean, logsigma, prior_mean=0, prior_logsigma=1):
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
        loss += kl.mean()
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
