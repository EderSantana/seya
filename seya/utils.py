import numpy as np
import theano
import theano.tensor as T
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
    return T.sqrt(T.sqr(z)+1e-6)


def unroll_scan(fn, sequences, outputs_info, non_sequences, n_steps,
                go_backwards=False):
        """
        Copy-pasted from Lasagne

        Helper function to unroll for loops. Can be used to unroll theano.scan.
        The parameter names are identical to theano.scan, please refer to here
        for more information.
        Note that this function does not support the truncate_gradient
        setting from theano.scan.
        Parameters
        ----------
        fn : function
            Function that defines calculations at each step.
        sequences : TensorVariable or list of TensorVariables
            List of TensorVariable with sequence data. The function iterates
            over the first dimension of each TensorVariable.
        outputs_info : list of TensorVariables
            List of tensors specifying the initial values for each recurrent
            value.
        non_sequences: list of TensorVariables
            List of theano.shared variables that are used in the step function.
        n_steps: int
            Number of steps to unroll.
        go_backwards: bool
            If true the recursion starts at sequences[-1] and iterates
            backwards.
        Returns
        -------
        List of TensorVariables. Each element in the list gives the recurrent
        values at each time step.
        """
        if not isinstance(sequences, (list, tuple)):
            sequences = [sequences]

        # When backwards reverse the recursion direction
        counter = range(n_steps)
        if go_backwards:
            counter = counter[::-1]

        output = []
        prev_vals = outputs_info
        for i in counter:
            step_input = [s[i] for s in sequences] + prev_vals + non_sequences
            out_ = fn(*step_input)
            # The returned values from step can be either a TensorVariable,
            # a list, or a tuple.  Below, we force it to always be a list.
            if isinstance(out_, T.TensorVariable):
                out_ = [out_]
            if isinstance(out_, tuple):
                out_ = list(out_)
            output.append(out_)

            prev_vals = output[-1]

        # iterate over each scan output and convert it to same format as scan:
        # [[output11, output12,...output1n],
        # [output21, output22,...output2n],...]
        output_scan = []
        for i in range(len(output[0])):
            l = map(lambda x: x[i], output)
            output_scan.append(T.stack(*l))

        return output_scan
