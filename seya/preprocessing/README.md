# Preprocessing

Preprocess data using external libraries.

### Eventual Dependencies
* natsort: `pip install natsort`
* Caffe
* [sklearn-theano](https://github.com/sklearn-theano/sklearn-theano)
* [imageio](https://github.com/imageio/imageio.git)

### Converting a video dataset to hdf5
Dependencies: `avconv` or `ffmpeg`, `h5py`.
Assume the following folder structure:
```
root
|-dataset
  |-video_class_1
    |-video_1.avi
    |-video_2.avi
  |-video_class_2
  ...
```

We want to convert all the videos to a single hdf5 tensor. That dataset will be
called `'data'` inside the hdf5. To do that,
we use the `convert_dataset_to_hdf5` that calls `convert_videos_to_hdf5`
recursively on each class folder. Also, on the same hdf5 we want a second matrix, called
`'labels'`, with the labels of each video given by the folder name. From the `root` directory we do the
following:
```python
from videos_to_hdf5 import convert_dataset_to_hdf5

def label_cbk(f, return_len=False):
    # define callback that calculates label from file name. If we don't
    # pass a label callback, the next function will only save the frames.
    if return_len:
        # label_callback must know the number of classes in the dataset
        return 51
    dirname = f.split('/')[1]
    names = [name for name in os.listdir('./dataset') if os.path.isdir(os.path.join('./dataset', name))]
    label = np.asarray([w == dirname for w in names])
    return label.astype('float32')
1
convert_dataset_to_hdf5('final_file.hdf5', 'dataset', ext='*.avi',
                        label_callback=label_cbk, convert='avconv')
```

The default configuration stores only the first 20 frames of each video center cropped
to 224x224.
If we need to filter the videos to generate train and test splits, it is easier to go through the hdf5 than
the original video dataset. In the example above, the videos order will be
saved to `final_file.hdf5.txt`.

Note: this code does not provide any exception treatment, use it at your
own risk or consider modifying it if you your dataset is too large.
