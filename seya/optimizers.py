import theano
import theano.tensor as T

from keras.optimizers import SGD, Adam
from keras.utils.theano_utils import shared_zeros


def _proxOp(x, l, soft=True):
    if soft:
        return T.maximum(x-l, 0) + T.minimum(x+l, 0)
    else:
        return T.switch(T.lt(abs(x),l), x, 0)


class ISTA(SGD):

    def __init__(self, lambdav=.1, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, soft_threshold=True, *args, **kwargs):
        super(ISTA, self).__init__(lr, momentum, decay, nesterov, *args,
                                   **kwargs)
        self.lambdav = lambdav
        self.soft_threshold = soft_threshold

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
            c_new_p = _proxOp(c(new_p), self.lr * self.lambdav,
                                   self.soft_threshold)
            self.updates.append((p, c_new_p))
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": self.lr,
                "lambdav": self.lambdav,
                "momentum": self.momentum,
                "decay": self.decay,
                "nesterov": self.nesterov}


class Adamista(Adam):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 lambdav=.0005, *args, **kwargs):
        super(Adamista, self).__init__(lr, beta_1, beta_2, epsilon, *args, **kwargs)
        self.lambdav = lambdav

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations+1.)]

        t = self.iterations + 1
        lr_t = self.lr * T.sqrt(1-self.beta_2**t)/(1-self.beta_1**t)

        for p, g, c in zip(params, grads, constraints):
            m = theano.shared(p.get_value() * 0.)  # zero init of moment
            v = theano.shared(p.get_value() * 0.)  # zero init of velocity

            m_t = (self.beta_1 * m) + (1 - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1 - self.beta_2) * (g**2)
            p_t = p - lr_t * m_t / (T.sqrt(v_t) + self.epsilon)

            c_p_t = _proxOp(c(p_t), self.lr * self.lambdav)
            self.updates.append((m, m_t))
            self.updates.append((v, v_t))
            self.updates.append((p, c_p_t))  # apply constraints
        return self.updates

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr.get_value()),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon}
