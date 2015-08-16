import theano
import numpy as np
floatX = theano.config.floatX


class TransformedDataset():
    def __init__(self, data, transformer):
        self.data = data
        self.transformer = transformer
        samp_out = transformer(data[:1])
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

    def transform(self, X):
        Rval = np.zeros((X.shape[0],) + self.shape[1:])
        for i, sample in enumerate(X):
            if len(self.img_shape) == 3:
                I = sample.reshape(self.img_shape).transpose(1, 2, 0)
            else:
                I = sample.reshape(self.img_shape)
            Rval[:, i, :] = self._allrotations(I, self.n_steps,
                                               self.img_shape,
                                               self.final_angle)
        Rval = Rval.astype(floatX)
        return Rval
