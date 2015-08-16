import theano
import numpy as np
import h5py

from collections import defaultdict
from skimage.transform import rotate
floatX = theano.config.floatX


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
                 time_start, time_end, normalizer=None):
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
            return self.normalizer(self.data[idx,
                                             self.time_start:self.time_end])
        else:
            return self.data[idx, self.time_start:self.time_end]

    @property
    def shape(self):
        return tuple((self.end - self.start, self.time_start -
                      self.time_end) + self.data.shape[2:])
