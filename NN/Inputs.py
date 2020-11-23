import os
import numpy as np
import h5py
import json
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
        kspace = k-space distribution of points that needs to be masked. Can be 2D or 3D.
        """
        mask = self.get_mask(kspace)
        return kspace * mask.astype(np.float) + 0.0

    def get_mask(self, kspace):
        """
        kspace = k-space distribution of points that needs to be masked. Can be 2D or 3D.
        """
        return self.rng.uniform(size=kspace.shape) < (1/self.acceleration)

class CenteredRandomMask(RandomMask):
    """
    Same as RandomMask but ensures center of kspace is fully sampled
    """
    def __init__(self,acceleration, center_fraction, seed=None):
        """
        docstring
        """
        self.acceleration = acceleration
        self.rng = np.random.RandomState()
        if seed:
            self.rng.seed(seed)
        self.center_fraction = center_fraction

    def get_mask(self, kspace):
        """
        kspace = k-space distribution of points that needs to be masked. Can be 2D.
        expected to have regular shape (shape[0]==shape[1])
        """
        #TODO code this better, generalise to ND
        size = kspace.shape[0]*kspace.shape[1]
        num_low_freqs = int(round(kspace.shape[0]*self.center_fraction))
        prob = (size/(size-(num_low_freqs**2)))/self.acceleration

        mask = self.rng.uniform(size=kspace.shape) < prob
        low = (kspace.shape[0] - num_low_freqs)/2
        high = (kspace.shape[0] + num_low_freqs)/2
        for i in range(kspace.shape[0]):
            for j in range(kspace.shape[1]):
                if i >= low and i<=high and j>=low and j<= high:
                    mask[i,j] = True
        return mask

def preprocess_data(path, name, cat, input_mask=None, output_mask=None, multicoil=False, normalise=False, fraction=None):
    """
    Prepares the work directory, and prepares data into easy-to-used, eventually masked data.
    """
    index_dict = {}
    files = [f for f in os.listdir(path) if (
        os.path.isfile(os.path.join(path, f)) and ('.h5' in f))]
    if fraction:
        files = files[:int(np.floor(fraction*len(files)))]
    if not os.path.isdir('D:\\NN_DATA\\{}'.format(name)):
        os.mkdir('D:\\NN_DATA\\{}'.format(name))
    if not os.path.isdir('D:\\NN_DATA\\{}\\{}'.format(name, cat)):
        os.mkdir('D:\\NN_DATA\\{}\\{}'.format(name, cat))
    for f in files:
        if os.path.isfile('D:\\NN_DATA\\{}\\{}\\{}'.format(name, cat, f)):
            continue
        h5f = h5py.File(os.path.join(path, f), 'r')
        if ('kspace' not in h5f):
            continue
        if ('reconstruction_esc' not in h5f):
            shape = (h5f['kspace'].shape[0], 320, 320)
        else:
            shape = h5f['reconstruction_esc'].shape
        inputs = np.empty(shape, dtype=np.float)
        outputs = np.empty(shape, dtype=np.float)
        index_dict['D:\\NN_DATA\\{}\\{}\\{}'.format(name, cat, f)] = h5f['kspace'].shape[0]
        for i, slic in enumerate(h5f['kspace']):
            inputs[i] = image_from_kspace(
                slic, multicoil=multicoil, mask=input_mask,normalise=normalise)
            outputs[i] = image_from_kspace(
                slic, multicoil=multicoil, mask=output_mask,normalise=normalise)
        h5f.close()
        outfile = h5py.File(
            'D:\\NN_DATA\\{}\\{}\\{}'.format(name, cat, f), 'w')
        outfile.create_dataset('inputs', data=inputs)
        outfile.create_dataset('outputs', data=outputs)
        outfile.close()
    if index_dict != {}:
        with open('D:\\NN_DATA\\{}\\{}\\index.json'.format(name, cat), 'w') as fp:
            json.dump(index_dict, fp)

def test_data(path,agent,batch_size=5):
    """
    Tests the agent on the data in the given path.
    """
    files = [f for f in os.listdir(path) if (
        os.path.isfile(os.path.join(path, f)) and ('.h5' in f))]
        
    for f in files:
        h5f = h5py.File(os.path.join(path, f), 'r+')
        inputs = np.reshape(h5f['inputs'],(*h5f['inputs'].shape,1))
        test_outputs = agent.test(inputs,batch_size=batch_size)
        h5f.create_dataset('tests',data=test_outputs)
        h5f.close()

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
        self.list_IDs = []
        with open('D:\\NN_DATA\\{}\\{}\\index.json'.format(name, cat), 'r') as fp:
            index_dict = json.load(fp)
            for fname in index_dict:
                for i in range(index_dict[fname]):
                    self.list_IDs.append([fname,i])
            fp.close()
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
