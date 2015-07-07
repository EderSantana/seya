from theano import tensor


def diff_abs(z):
    return tensor.sqrt(tensor.sqr(z)+1e-6)
