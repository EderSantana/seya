from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers.core import MaskedLayer, Layer, ActivityRegularization
from keras.layers.convolutional import MaxPooling2D

from seya.regularizers import LambdaRegularizer


class MaskedBN(MaskedLayer, BatchNormalization):
    def __init__(self, *args, **kwargs):
        super(MaskedBN, self).__init__(*args, **kwargs)
