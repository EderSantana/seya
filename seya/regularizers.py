import theano.tensor as T
from keras.regularizers import Regularizer


class GaussianKL(Regularizer):
    """ KL-divergence between two gaussians.
    Useful for Variational AutoEncoders.
    Use this as an activation regularizer
    """
    def set_param(self, p):
        self.p = p

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        # See Variational Auto-Encoding Bayes by Kingma and Welling.
        mean, logsigma = self.layer.get_output(True)
        kl = -.5 - logsigma + .5 * (mean**2
                                    + T.exp(2 * logsigma))
        loss += kl.mean()
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__}


class SimpleCost(Regularizer):
    def __init__(self, cost):
        super(SimpleCost, self).__init__()
        self.cost = cost

    def __call__(self, loss):
        loss += self.cost
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__}
