import os
import numpy as np
import h5py
import random

from PIL import Image
from glob import glob
from sklearn_theano.feature_extraction import OverfeatTransformer as fext

from keras.utils.generic_utils import Progbar


def features_to_hdf5(input_path, output_path='features.hdf5',
                     feature_extractor=fext, label_callback=None,
                     img_size=(231, 231)):
    ''' Extract features using a pretrained model and store the results to hdf5.

    Parameteres:
    ------------
    input_path: str, path to images with appropriate extensions example
                "./path/*.jpg" or "./path/*.png"

    output_path: str, path to the resulting hdf5 file. We also create a text
                 file with the name and order of the files saved to hdf5.

    label_callback: optional function, a callback to get a label from each file
                    name. Should return a numpy array with labels, even when
                    using a single binary label. We use the `len` of the
                    returned value to allocate the labels matrix.

    feature_extractor: sklearn-theano transformer, default: OverfeatTransformer

    img_size: list (int, int): number of rows and columns for the converted
              image

    Notes:
    ------
    Input files are naturally sorted with natsort and features are saved on
    that order.

    This script creates a temporary file called tttemp.hdf5 and deletes
    afterward.

    This code converts one image at a time, for
    really large datasets you may want to modify it to convert images in batches.
    '''
    files = glob(input_path)
    random.shuffle(files)
    output_txt = file(output_path + '.txt', 'w')
    tf = fext(output_layers=(-3,))

    count = 0

    with h5py.File('tttemp.hdf5', 'w') as h:
        feature_size = _load_and_convert(files[0], tf, img_size).shape[-1]
        X = h.create_dataset('features', (len(files), feature_size), dtype='f')
        if label_callback is not None:
            label_size = len(label_callback('test'))
            y = h.create_dataset('labels', (len(files), label_size), dtype='f')

        print("Extracting features of a total of {} files...".format(len(files)))
        progbar = Progbar(len(files))
        for f in files:
            try:
                # Some of these files are not reading right...
                im2 = _load_and_convert(f, tf, img_size)
                X[count] = im2
                if label_callback is not None:
                    y[count] = label_callback(f)
                count += 1
                output_txt.write(f + "\n")
                if count % 100 == 0:
                    progbar.add(100)

            except:
                print("Problem converting: {}".format(f))
                continue

        print("Successfully transformed {} files".format(count))
        with h5py.File(output_path, 'w') as g:
            XX = g.create_dataset('features', (count, feature_size), dtype='f')
            if label_callback is not None:
                yy = g.create_dataset('labels', (count, label_size), dtype='f')
                yy[:] = y[:count]

            progbar = Progbar(count)
            print("Cleaning up...")
            for i in range(count):
                progbar.add(1)
                XX[i] = X[i]
    del tf
    output_txt.close()
    os.system('rm tttemp.hdf5')


def _load_and_convert(f, tf, img_size):
    im = Image.open(f, 'r')
    I = np.array(im.resize(img_size))
    return tf.transform(I[np.newaxis])[0]
