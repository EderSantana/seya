import theano.tensor as T

from keras.optimizers import SGD
from keras.utils.theano_utils import shared_zeros


class ISTA(SGD):

    def __init__(self, lambdav=.1, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, *args, **kwargs):
        super(ISTA, self).__init__(lr, momentum, decay, nesterov, *args,
                                   **kwargs)
        self.lambdav = lambdav

    def _proxOp(self, x, l):
        return T.maximum(x-l, 0) + T.minimum(x+l, 0)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        self.updates = [(self.iterations, self.iterations + 1.)]

        for p, g, c in zip(params, grads, constraints):
            m = shared_zeros(p.get_value().shape)  # momentum
            v = self.momentum * m - lr * g  # velocity
            self.updates.append((m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v
            c_new_p = self._proxOp(c(new_p), self.lr * self.lambdav)
            self.updates.append((p, c_new_p))
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": self.lr,
                "lambdav": self.lambdav,
                "momentum": self.momentum,
                "decay": self.decay,
                "nesterov": self.nesterov}
