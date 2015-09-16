import numpy as np
import theano.tensor as tensor
from keras.layers.core import Activation
from theano.sandbox.rng_mrg import MRG_RandomStreams


def alloc_ones_matrix(*dims):
    return T.alloc(np.cast[theano.config.floatX](1.), *dims)


def theano_rng(seed=123):
    return MRG_RandomStreams(seed=seed)


def apply_layer(layer, X):
    flag = False
    try:
        tmp = layer.input
        flag = True
    except:
        pass
    if isinstance(layer, Activation):
        return layer.activation(X)
    else:
        layer.input = X
        Y = layer.get_output()
        if flag:
            layer.input = tmp
        return Y


def apply_model(model, X):
    tmp = model.layers[0].input
    model.layers[0].input = X
    Y = model.get_output()
    model.layers[0].input = tmp
    return Y


def s2s_to_s2t(sequences, targets):
    '''
    Transforms as sequence to sequence problem to a sequence to target.
    It does so by replicating the input dataset a lot.
    So use this only with small datasets.
    Also, there is no way of passing the hidden states from one batch to another,
    thus this is not the best way to solve this problem.
    '''
    X = []
    Xrev = []  # reversed dataset
    y = []
    for seq, tar in zip(sequences, targets):
        if not len(seq) == len(tar):
            raise ValueError("Sequences and Targets must have the same length.")
        for i in range(len(seq)):
            X.append(seq[:i+1])
            if i == 0:
                Xrev.append(seq[::-1])
            else:
                Xrev.append(seq[:i-1:-1])
            # X.append(seq[:i+1])
            # Xrev.append(seq[:i-1:-1])
            y.append(tar[i])
    return X, Xrev, y


def pad_md_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    '''Pad multi-dimensional sequence
    Similar to keras.preprocesing.sequence but supports a third dimension:

        Pad each sequence to the same length:
        the length of the longuest sequence.

        If maxlen is provided, any sequence longer
        than maxlen is truncated to maxlen. Truncation happens off either the beginning (default) or
        the end of the sequence.

        Supports post-padding and pre-padding (default).

    '''
    lengths = [len(s) for s in sequences]
    dim = sequences[0].shape[-1]  # the third dimension is assumed equal for all elements

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen, dim)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def diff_abs(z):
    return tensor.sqrt(tensor.sqr(z)+1e-6)
