from __future__ import print_function

# import imageio
import os
import h5py
from glob import glob

import subprocess
from scipy.misc import imread, imresize


def convert_videos_to_hdf5(hdf5file, filepath,
                           ext='*.avi',
                           img_shape=(224, 224),
                           frame_range=range(0, 20),
                           row_range=slice(120-112, 120+112),
                           col_range=slice(160-112, 160+112),
                           label_callback=None,
                           converter='avconv'):

    """
    Convert all the videos of a folder to images and dump the images to an hdf5
    file.

    Parameters:
    -----------
    hdf5file: str, path to output hdf5 file
    filepath: str, path to folder with videos
    ext: str, video extensions (default *.avi)
    img_shape: tuple, (row, col) size of the image crop of video
    frame_range: list, desired frames of the video
    row_range: slice, slice of the image rows
    col_rance: slice, slice of the image cols

    Results:
    --------
    An hdf5 file with videos stored as array

    """
    rlen = row_range.stop - row_range.start
    clen = col_range.stop - col_range.start
    files = glob(os.path.join(filepath, ext))
    with h5py.File(hdf5file, 'w') as h5:
        # create datasets
        X = h5.create_dataset('data', (len(files), len(frame_range),
                              rlen, clen, 3), dtype='f')
        if label_callback is not None:
                label_size = label_callback('test', return_len=True)
                y = h5.create_dataset('labels', (len(files), label_size), dtype='f')

        for c1, f in enumerate(files):
            process = subprocess.Popen('mkdir {}'.format(f[:-4]), shell=True, stdout=subprocess.PIPE)
            process.wait()
            cmd = converter + " -i {0} -r 1/1 {1}/%03d.jpg".format(f, f[:-4])
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            process.wait()

            imgs = glob(os.path.join(f[:-4], '*.jpg'))
            for c2, im in enumerate(imgs[:frame_range[-1]]):
                I = imread(im)[row_range, col_range]
                if I.shape[:2] != img_shape:
                    I = imresize(imread(im), img_shape)
                X[c1, c2] = I

            if label_callback is not None:
                y[c1] = label_callback(f)

            process = subprocess.Popen('rm {}'.format(f[:-4]), shell=True, stdout=subprocess.PIPE)
            process.wait()

    with open(hdf5file+'.txt', 'w') as file_order:
        for f in files:
            file_order.write(f+'\n')


def convert_dataset_to_hdf5(hdf5file, filepath,
                            ext='*.avi',
                            img_shape=(224, 224),
                            frame_range=range(0, 20),
                            row_range=slice(120-112, 120+112),
                            col_range=slice(160-112, 160+112),
                            label_callback=None):
    """
    Convert all the videos of a dataset to images and dump the images to an hdf5
    file. Uses convert_videos_to_hdf5 recursively on folder deep in a file
    structure.

    Parameters:
    -----------
    hdf5file: str, path to output hdf5 file
    filepath: str, path to folder with videos
    ext: str, video extensions (default *.avi)
    img_shape: tuple, (row, col) size of the image crop of video
    frame_range: list, desired frames of the video
    row_range: slice, slice of the image rows
    col_rance: slice, slice of the image cols

    Results:
    --------
    An hdf5 file with videos stored as array

    """
    dirs = [d for d in glob(os.path.join(filepath, '*')) if os.path.isdir(d)]
    files = glob(os.path.join(filepath, '*', ext))
    rlen = row_range.stop - row_range.start
    clen = col_range.stop - col_range.start

    process = subprocess.Popen('mkdir hdf5_files_temp', shell=True, stdout=subprocess.PIPE)
    process.wait()

    start = 0
    with h5py.File(hdf5file, 'w') as h5:
        # create datasets
        X = h5.create_dataset('data', (len(files), len(frame_range),
                              rlen, clen, 3), dtype='f')
        if label_callback is not None:
            label_size = label_callback('test', return_len=True)
            y = h5.create_dataset('labels', (len(files), label_size), dtype='f')

        print('Converting {0} directories, {1} files'.format(len(dirs), len(files)))
        for i, d in enumerate(dirs):
            print('Converting: {0}/{1} {2}'.format(i+1, len(dirs), d))
            convert_videos_to_hdf5('hdf5_files_temp.h5', d,
                                   ext,
                                   img_shape,
                                   frame_range,
                                   row_range,
                                   col_range,
                                   label_callback)
            with h5py.File('hdf5_files_temp.h5', 'r') as temp:
                n_samples = temp['data'].shape[0]
                X[i+start:i+start+n_samples] = temp['data']
                if label_callback is not None:
                    y[i+start:i+start+n_samples] = temp['labels']
                start += n_samples
'''
WORK IN PROGRESSS: waiting for imageio to fix some issues

def videos_to_hdf5(hdf5file, videospath,
                   label_callback=None,
                   ext='*.avi',
                   frame_range=range(0, 20),
                   row_range=slice(120-112, 120+112),
                   col_range=slice(160-112, 160+112)):
    """
    Convert all the videos of a nested folder directory of videos

    Problems:
    ---------
    imageio has problems reading too many videos.

    """
    rlen = row_range.stop - row_range.start
    clen = col_range.stop - col_range.start
    files = glob(os.path.join(videospath, '*', ext))
    with h5py.File(hdf5file, 'w') as h5:
        # create datasets
        X = h5.create_dataset('data', (len(files), len(frame_range),
                                       rlen, clen, 3), dtype='f')

        cPickle.dump(files, file('filename.pkl', 'w'), -1)

        if label_callback is not None:
            label_size = len(label_callback('test'))
            y = h5.create_dataset('labels', (len(files), label_size), dtype='f')
        # read files
        for c, f in enumerate(files):
            with imageio.get_reader(f,  'ffmpeg') as vid:
                for i, fr in enumerate(frame_range):
                    X[c, i] = vid.get_data(fr)[row_range, col_range].astype('float')
                    if label_callback is not None:
                        y[c] = label_callback(f)
'''
