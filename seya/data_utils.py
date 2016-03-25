import theano
import numpy as np
import h5py
import threading

from keras.datasets import mnist
from keras.utils import np_utils

from collections import defaultdict
from skimage.transform import rotate
floatX = theano.config.floatX


def load_rotated_mnist():
    (_, y_train), (_, y_test) = mnist.load_data()

    X_train = HDF5Tensor('/home/eders/python/blog/rotated_mnist_train.hdf5', 'X_train', 0, 50000, 0, 19)
    X2_train = HDF5Tensor('/home/eders/python/blog/rotated_mnist_train.hdf5', 'X_train', 0, 50000, 1, 20)

    X_valid = HDF5Tensor('/home/eders/python/blog/rotated_mnist_train.hdf5', 'X_valid', 0, 10000-16, 0, 19)
    X2_valid = HDF5Tensor('/home/eders/python/blog/rotated_mnist_train.hdf5', 'X_valid', 0, 10000-16, 1, 20)

    X_test = HDF5Tensor('/home/eders/python/blog/rotated_mnist_train.hdf5', 'X_test', 0, 10000-16, 0, 19)
    X2_test = HDF5Tensor('/home/eders/python/blog/rotated_mnist_train.hdf5', 'X_test', 0, 10000-16, 1, 20)

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    return (X_train, X2_train, Y_train[:50000],
            X_valid, X2_valid, Y_train[50000:],
            X_test, X2_test, Y_test)


class TransformedDataset():
    def __init__(self, data, transformer):
        self.data = data
        self.transformer = transformer
        samp_out = transformer.transform(data[:2])
        self.shape = (data.shape[0],) + samp_out.shape[1:]

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        b = self.data.__getitem__(key)
        b = self.transformer.transform(b)
        return b


class DataTransformer():
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self, X):
        pass


class RotateData(DataTransformer):
    '''Generate rotated images.
       This functions is supposed to be used with
       `seya.data_utils.TransformedDataset`

       Expected data shape batch_size x dim
    '''
    def __init__(self, n_steps, img_shape=(28, 28), final_angle=180):
        self.n_steps = n_steps
        self.img_shape = img_shape
        self.final_angle = final_angle

    def _allrotations(self, image):
        angles = np.linspace(0, self.final_angle, self.n_steps)
        R = np.zeros((self.n_steps, np.prod(self.img_shape)))
        for i in xrange(self.n_steps):
            img = rotate(image, angles[i])
            if len(self.img_shape) == 3:
                img = img.transpose(2, 0, 1)
            R[i] = img.flatten()
        return R

    def transform(self, X):
        Rval = np.zeros((X.shape[0], self.n_steps, X.shape[1]))
        for i, sample in enumerate(X):
            if len(self.img_shape) == 3:
                I = sample.reshape(self.img_shape).transpose(1, 2, 0)
            else:
                I = sample.reshape(self.img_shape)
            Rval[i] = self._allrotations(I)
        Rval = Rval.astype(floatX)
        return Rval


class HDF5Tensor():
    def __init__(self, datapath, dataset, start, end,
                 time_start, time_end, time_step=1, normalizer=None):
        self.refs = defaultdict(int)
        if datapath not in list(self.refs.keys()):
            f = h5py.File(datapath)
            self.refs[datapath] = f
        else:
            f = self.refs[datapath]
        self.start = start
        self.end = end
        self.data = f[dataset]
        self.normalizer = normalizer
        self.time_start = time_start
        self.time_end = time_end
        self.time_step = time_step

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.stop + self.start <= self.end:
                idx = slice(key.start+self.start, key.stop + self.start)
            else:
                raise IndexError
        elif isinstance(key, int):
            if key + self.start < self.end:
                idx = key+self.start
            else:
                raise IndexError
        elif isinstance(key, np.ndarray):
            if np.max(key) + self.start < self.end:
                idx = (self.start + key).tolist()
            else:
                raise IndexError
        elif isinstance(key, list):
            if max(key) + self.start < self.end:
                idx = [x + self.start for x in key]
            else:
                raise IndexError
        if self.normalizer is not None:
            return self.normalizer(
                self.data[idx, self.time_start:self.time_end:self.time_step])
        else:
            return self.data[idx, self.time_start:self.time_end:self.time_step]

    @property
    def shape(self):
        return tuple((self.end - self.start,
                      (self.time_end - self.time_start)/self.time_step) +
                     self.data.shape[2:])

    @property
    def ndim(self):
        return len(self.data.shape)


class IndexedGenerator(object):
    """IndexedGenerator
    Handles datasets with arbitrary dataset indeces

    Usage:
    ------
    Assume that you have a list of valid indices ids = [0, 10, 11]
    Create `datagen = IndexedH5(ids)`
    Define data `X = h5py.File("mydataset.h5")['dataset']  # works even with HDF5`
    Fit model `model.fit_generator(datagen.flow(X, Labels, batch_size=32),
                                   samples_per_epoch=len(ids),
                                   nb_epoch=1, show_accuracy=True,
                                   nb_worker=8))

    Parameters:
    -----------
    indices: array of int, numpy array of dataset indices

    """
    def __init__(self, indices=None, callback=None):
        if indices is not None:
            self.indices = indices
        else:
            self.indices = range(len(indices))
        # self.data = f[dataset]
        self.callback = callback
        self.lock = threading.Lock()

    def _flow_index(self, N, batch_size=32, shuffle=True, seed=None):
        b = 0
        total_b = 0
        while 1:
            if b == 0:
                if seed is not None:
                    np.random.seed(seed + total_b)

                if shuffle:
                    index_array = np.random.shuffle(self.indices)
                else:
                    index_array = np.arange(len(self.indices))

            current_index = (b * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
            else:
                current_batch_size = N - current_index

            if current_batch_size == batch_size:
                b += 1
            else:
                b = 0
            total_b += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def flow(self, X, y, batch_size=32, shuffle=False, seed=None):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.flow_generator = self._flow_index(self.indices.shape[0], batch_size,
                                               shuffle, seed)
        return self

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see # http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.flow_generator)
        idx = sorted(self.indices[index_array.tolist()].tolist())
        # import pdb; pdb.set_trace()
        bX = self.X[idx]
        if self.callback is not None:
            bX = self.callback(bX)
        bY = self.y[idx]
        return bX, bY

    def __next__(self):
        # for python 3.x.
        return self.next()
