import os
import numpy as np
import h5py
import pickle
from tensorflow.keras.utils import Sequence
from utils.fastMRI_utils import image_from_kspace

# TODO use mongoDB to handle file locations


class RandomMask(object):
    """
    Class that holds and applies a completely random k-space mask on images.
    """

    def __init__(self, acceleration, seed=None):
        """
        acceleration = acceleration factor for the masking, if the acceleration is 2, then half the k-space points will not be masked, if it is 5 then only 20 % of the points will not be masked.
        seed = rng seed for reproducibility.
        """
        self.acceleration = acceleration
        self.rng = np.random.RandomState()
        if seed:
            self.rng.seed(seed)

    def __call__(self, kspace):
        """
        kspace = k-space distribution of points that needs to be maskes. Can be 2D or 3D.
        """
        mask = self.rng.uniform(size=kspace.shape) < (1/self.acceleration)
        return kspace * mask.astype(np.float) + 0.0


def preprocess_data(path, name, cat, input_mask=None, output_mask=None, multicoil=False):
    """
    Prepares the work directory, and prepares data into easy-to-used, eventually masked data.
    """
    index_list = []
    files = [f for f in os.listdir(path) if (
        os.path.isfile(os.path.join(path, f)) and ('.h5' in f))]
    if not os.path.isdir('D:\\NN_DATA\\{}'.format(name)):
        os.mkdir('D:\\NN_DATA\\{}'.format(name))
    if not os.path.isdir('D:\\NN_DATA\\{}\\{}'.format(name, cat)):
        os.mkdir('D:\\NN_DATA\\{}\\{}'.format(name, cat))
    for f in files:
        if os.path.isfile('D:\\NN_DATA\\{}\\{}\\{}'.format(name, cat, f)):
            continue
        h5f = h5py.File(os.path.join(path, f), 'r')
        if ('kspace' not in h5f) or ('reconstruction_esc' not in h5f):
            continue
        inputs = np.empty(h5f['reconstruction_esc'].shape, dtype=np.float)
        outputs = np.empty(h5f['reconstruction_esc'].shape, dtype=np.float)
        for i, slic in enumerate(h5f['kspace']):
            index_list.append(
                ['D:\\NN_DATA\\{}\\{}\\{}'.format(name, cat, f), i])
            inputs[i] = image_from_kspace(
                slic, multicoil=multicoil, mask=input_mask)
            outputs[i] = image_from_kspace(
                slic, multicoil=multicoil, mask=output_mask)
        h5f.close()
        outfile = h5py.File(
            'D:\\NN_DATA\\{}\\{}\\{}'.format(name, cat, f), 'w')
        outfile.create_dataset('inputs', data=inputs)
        outfile.create_dataset('outputs', data=outputs)
        outfile.close()
    with open('D:\\NN_DATA\\{}\\{}\\index.pck'.format(name, cat), 'wb') as fp:
        pickle.dump(index_list, fp)


class DataGenerator(Sequence):
    """
    Generates data for Keras.
    """

    def __init__(self, name, cat, batch_size=32, dim=(320, 320), n_channels=1, shuffle=True):
        """
        Initialization.
        """
        self.dim = dim
        self.batch_size = batch_size
        with open('D:\\NN_DATA\\{}\\{}\\index.pck'.format(name, cat), 'rb') as fp:
            self.list_IDs = pickle.load(fp)
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples.
        """
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim,
                      self.n_channels), dtype=np.float)
        y = np.empty((self.batch_size, *self.dim,
                      self.n_channels), dtype=np.float)

        # Generate data
        f = h5py.File(list_IDs_temp[0][0], 'r')
        for i, ID in enumerate(list_IDs_temp):
            # open right file if needed
            if f.filename != ID[0]:
                f.close()
                f = h5py.File(ID[0], 'r')
            X[i] = np.reshape(f['inputs'][ID[1]], (*self.dim, self.n_channels))
            y[i] = np.reshape(f['outputs'][ID[1]],
                              (*self.dim, self.n_channels))
        f.close()

        return X, y
