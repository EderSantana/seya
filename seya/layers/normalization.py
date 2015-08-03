from keras.layers.normalization import BatchNormalization
from keras.layers.core import MaskedLayer


class MaskedBN(MaskedLayer, BatchNormalization):
    def __init__(self, *args, **kwargs):
        super(MaskedBN, self).__init__(*args, **kwargs)
