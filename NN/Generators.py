import os
import numpy as np
import json
import h5py
from tensorflow.keras.utils import Sequence

class BaseDataGenerator(Sequence):
    """
    Base class for data generators, holds the basic common functions.
    """
    
    def __init__(self, datadir_path, batch_size=8, shuffle=True):
        """
        Initialization.
        """
        self.batch_size = batch_size
        self.list_IDs = []
        with open(os.path.join(datadir_path,'index.json'), 'r') as fp:
            index_dict = json.load(fp)
            for fname in index_dict:
                for i in range(index_dict[fname]):
                    self.list_IDs.append([fname,i])
            fp.close()
        with open(os.path.join(datadir_path,'format.json'), 'r') as fp:
            self.data_shape = json.load(fp)
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


class DataGenerator_kspace_img(BaseDataGenerator):
    """
    Generates data for a Wnet_like architecture.
    """
    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples.
        """
        X = np.empty((self.batch_size, *self.data_shape))
        intermediate_y = np.empty((self.batch_size, *self.data_shape))
        y = np.empty((self.batch_size, *self.data_shape))

        # Generate data
        f = h5py.File(list_IDs_temp[0][0], 'r')
        for i, ID in enumerate(list_IDs_temp):
            # open right file if needed
            if f.filename != ID[0]:
                f.close()
                f = h5py.File(ID[0], 'r')
            X[i] = f['kspace_masked'][ID[1]]
            intermediate_y[i] = f['image_ground_truth'][ID[1]]
            y[i] = f['image_ground_truth'][ID[1]]
        f.close()
        return X, [intermediate_y,y]

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

class DataGenerator_kspace_to_img(BaseDataGenerator):
    """
    Generates data for a Wnet_like architecture.
    """
    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples.
        """
        X = np.empty((self.batch_size, *self.data_shape))
        y = np.empty((self.batch_size, *self.data_shape))

        # Generate data
        f = h5py.File(list_IDs_temp[0][0], 'r')
        for i, ID in enumerate(list_IDs_temp):
            # open right file if needed
            if f.filename != ID[0]:
                f.close()
                f = h5py.File(ID[0], 'r')
            X[i] = f['kspace_masked'][ID[1]]
            y[i] = f['image_ground_truth'][ID[1]]
        f.close()
        return X, y

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

class DataGenerator_img(BaseDataGenerator):
    """
    Generates data for a Wnet_like architecture.
    """
    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples.
        """
        X = np.empty((self.batch_size, *self.data_shape))
        y = np.empty((self.batch_size, *self.data_shape))

        # Generate data
        f = h5py.File(list_IDs_temp[0][0], 'r')
        for i, ID in enumerate(list_IDs_temp):
            # open right file if needed
            if f.filename != ID[0]:
                f.close()
                f = h5py.File(ID[0], 'r')
            X[i] = f['image_masked'][ID[1]]
            y[i] = f['image_ground_truth'][ID[1]]
        f.close()
        return X,y

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

class DataGenerator_kspace(BaseDataGenerator):
    """
    Generates data for a Wnet_like architecture.
    """
    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples.
        """
        X = np.empty((self.batch_size, *self.data_shape))
        y = np.empty((self.batch_size, *self.data_shape))

        # Generate data
        f = h5py.File(list_IDs_temp[0][0], 'r')
        for i, ID in enumerate(list_IDs_temp):
            # open right file if needed
            if f.filename != ID[0]:
                f.close()
                f = h5py.File(ID[0], 'r')
            X[i] = f['kspace_masked'][ID[1]]
            y[i] = f['kspace_ground_truth'][ID[1]]
        f.close()
        return X,y
        
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


generator_dict = {
    'imgtoimg':DataGenerator_img,
    'kspacetokspace':DataGenerator_kspace,
    'kspacetoimg':DataGenerator_kspace_to_img,
    'kspacetokspace&img':DataGenerator_kspace_img
}

def find_generator(path):
    if 'Wnet' in path:
        return generator_dict['kspacetokspace&img']
    elif 'kspacetoimg' in path:
        return generator_dict['kspacetoimg']
    elif 'img' in path:
        return generator_dict['imgtoimg']
    elif 'kspace' in path:
        return generator_dict['kspacetokspace']
    else:
        print('unknown generator format case')
        import pdb;pdb.set_trace()