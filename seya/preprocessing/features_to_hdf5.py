from PIL import Image
from glob import glob
from sklearn_theano.feature_extraction import OverfeatTransformer as fext
import h5py
from natsort import natsorted
import os
import numpy as np


def image2features(input_path, output_path='features.hdf5',
                   feature_extractor=fext, label_callback=None):
    ''' Extract features using a pretrained model and store to hdf5

    Parameteres:
    ------------
    input_path: str, path to images with appropriate extensions example
                "./path/*.jpg" or "./path/*.png"

    output_path: str, path to the resulting hdf5 file

    label_callback: optional function, a callback to get label from each file
                    name

    feature_extractor: sklearn-theano transformer, default: OverfeatTransformer

    Notes:
    ------
    Input files are naturally sorted with natsort and features are saved on
    that order.

    This script creates a temporary file called tttemp.hdf5 and deletes
    afterward.
    '''
    files = natsorted(glob(input_path))
    tf = fext(output_layers=(-3,))

    count = 0

    with h5py.File('tttemp.hdf5', 'w') as h:
        X = h.create_dataset('features', (len(files), 3072), dtype='f')
        if label_callback is not None:
            label_size = len(label_callback('test'))
            y = h.create_dataset('labels', (len(files), label_size), dtype='f')

        print("Reading a total of {} files...".format(len(files)))
        for f in files:
            try:
                # Some of these files are not reading right...
                im = Image.open(f, 'r')
                im
                I = np.array(im.resize((231, 231)))
                im2 = tf.transform(I[np.newaxis])[0]
                X[count] = im2
                if label_callback is not None:
                    y[count] = label_callback(f)
                count += 1
                print("Converted: {}".format(f))
            except:
                print("Problem loading: {}".format(f))
                continue

        print("Successfully transformed {} files".format(count))
        with h5py.File(output_path, 'w') as g:
            XX = g.create_dataset('features', (count, 3072), dtype='f')
            if label_callback is not None:
                yy = g.create_dataset('labels', (count, label_size), dtype='f')
                yy[:] = y[:count]
            for i in range(count):
                XX[i] = X[i]
    del tf
    os.system('rm tttemp.hdf5')
