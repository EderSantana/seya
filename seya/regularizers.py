import theano.tensor as T
from keras.regularizers import Regularizer


class GaussianKL(Regularizer):
    def set_param(self, p):
        self.p = p

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        # See Variational Auto-Encoding Bayes by Kingma and Welling.
        mean, sigma = self.layer.get_output(True)
        kl = -.5 - self.logsigma + .5 * (mean**2
                                         + T.exp(2 * self.logsigma))
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
