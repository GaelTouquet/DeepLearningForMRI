import os
import numpy as np
import h5py
import json
from tensorflow.keras.utils import Sequence
from utils.fastMRI_utils import image_from_kspace, crop, fft, ifft

# import matplotlib.pyplot as plt

# TODO use mongoDB to handle file locations

def prepare_image(kdata,crop_size=(320,320), normalise=True,mask=None):

    if normalise:
        kdata*=1.e6

    image = ifft(kdata)

    if crop:
        image_clean = crop(image,size=crop_size)
        # if normalise:
        #     image_clean *= 1./np.amax(image)
        kdata = fft(image_clean)
    
    kdata_clean = kdata

    if mask:
        kdata = mask(kdata_clean)
    else:
        kdata = kdata_clean
    
    # fig, axs = plt.subplots(2,3, figsize=(20, 20))
    # fig.subplots_adjust(hspace=0.1, wspace=0.1)
    # axs = axs.ravel()
    # axs[0].imshow(np.real(ifft(kdata)),cmap='gray')
    # axs[1].imshow(np.real(ifft(kdata_clean)),cmap='gray')
    # axs[2].imshow(np.real(image_clean),cmap='gray')
    # axs[3].imshow(np.imag(ifft(kdata)),cmap='gray')
    # axs[4].imshow(np.imag(ifft(kdata_clean)),cmap='gray')
    # axs[5].imshow(np.imag(image_clean),cmap='gray')
    # plt.show()

    return kdata, kdata_clean, image_clean
    

def prepare_datasets(datapath, workdirpath, dataset_type, 
    input_mask=None, output_mask=None,
    multicoil=False, normalise=False, 
    fraction=None, n_slice_per_file=None,
    input_shape=(320,320),
    intermediate_shape=None,
    output_shape=None,
    n_channels_in=2,
    n_channels_intermediate=2,
    n_channels_out=2):
    """
    Prepares the work directory, and prepares data into easy-to-used, eventually masked data.
    """
    if intermediate_shape is None:
        intermediate_shape = input_shape
    if output_shape is None:
        output_shape = input_shape


    index_dict = {}

    files = [f for f in os.listdir(datapath) if (
        os.path.isfile(os.path.join(datapath, f)) and ('.h5' in f))]
    if fraction:
        files = files[:int(np.floor(fraction*len(files)))]

    output_dir = os.path.join(workdirpath,dataset_type)
    if not os.path.isdir(output_dir):
         os.mkdir(output_dir)

    for f in files:
        filepath = os.path.join(output_dir,f)
        if os.path.isfile(filepath):
            continue
        h5f = h5py.File(os.path.join(datapath, f), 'r')
        if ('kspace' not in h5f):
            continue

        if n_slice_per_file is None:
            n_slices = h5f['kspace'].shape[0]
        else:
            n_slices = n_slice_per_file
        # inputs = np.empty((h5f['kspace'].shape[0], *image_shape,2), dtype=np.float)
        # outputs = np.empty((h5f['kspace'].shape[0], *image_shape,2), dtype=np.float)#carefull here 1 channel for abs(cplx)
        # intermediate_outputs = np.empty((h5f['kspace'].shape[0], *image_shape,2), dtype=np.float)
        # index_dict['D:\\NN_DATA\\{}\\{}\\{}'.format(name, cat, f)] = h5f['kspace'].shape[0]
        inputs = np.empty((n_slices, *input_shape,n_channels_in), dtype=np.float)
        intermediate_outputs = np.empty((n_slices, *intermediate_shape,n_channels_intermediate), dtype=np.float)
        outputs = np.empty((n_slices, *output_shape,n_channels_out), dtype=np.float)#carefull here 1 channel for abs(cplx)
        index_dict[filepath] = n_slices

        k = 0
        imin = int(np.floor(h5f['kspace'].shape[0]/2 - n_slices/2))
        imax = imin + n_slices
        for i, slic in enumerate(h5f['kspace']):
            if i<imin or i>=imax:
                continue
            kdata, kdata_clean, image = prepare_image(slic, crop_size=input_shape,normalise=normalise,mask=input_mask)
            inputs[k,:,:,0] = np.real(kdata)
            inputs[k,:,:,1] = np.imag(kdata)

            intermediate_outputs[k,:,:,0] = np.abs(image)
            intermediate_outputs[k,:,:,1] = np.angle(image)
            # outputs[k,:,:,0] = np.abs(image)
            if n_channels_out ==1:
                outputs[k,:,:,0] = np.abs(image)
            else:
                outputs[k,:,:,0] = np.abs(image)
                outputs[k,:,:,1] = np.angle(image)
            k+=1
        h5f.close()
        outfile = h5py.File(filepath,'w')
        outfile.create_dataset('inputs', data=inputs)
        outfile.create_dataset('intermediate_outputs', data=intermediate_outputs)
        outfile.create_dataset('outputs', data=outputs)
        outfile.close()
    indexfile_path = os.path.join(output_dir,'index.json')
    if index_dict!={}:
        with open(indexfile_path, 'w') as fp:
            json.dump(index_dict, fp)
    return indexfile_path

def test_data(path,agent,batch_size=5):
    """
    Tests the agent on the data in the given path.
    """
    files = [f for f in os.listdir(path) if (
        os.path.isfile(os.path.join(path, f)) and ('.h5' in f))]
        
    for f in files:
        h5f = h5py.File(os.path.join(path, f), 'r+')
        inputs = np.reshape(h5f['inputs'],h5f['inputs'].shape)#np.reshape(h5f['inputs'],(h5f['inputs'].shape[0],*indim))
        test_outputs = agent.test(inputs,batch_size=batch_size)
        h5f.create_dataset('tests',data=test_outputs)
        h5f.close()

class DataGenerator(Sequence):
    """
    Generates data for Keras.
    """

    def __init__(self, indexfile_path, batch_size=32, indim=(320, 320,2), outdim=(320, 320,2), n_channels_in=1, n_channels_out=1, shuffle=True):
        """
        Initialization.
        """
        self.indim = indim
        self.outdim = outdim
        self.batch_size = batch_size
        self.list_IDs = []
        with open(indexfile_path, 'r') as fp:
            index_dict = json.load(fp)
            for fname in index_dict:
                for i in range(index_dict[fname]):
                    self.list_IDs.append([fname,i])
            fp.close()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
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
        X = np.empty((self.batch_size, *self.indim,
                      self.n_channels_in), dtype=np.float)
        intermediate_y = np.empty((self.batch_size, *self.indim,
                      self.n_channels_in), dtype=np.float)
        y = np.empty((self.batch_size, *self.outdim,
                      self.n_channels_out), dtype=np.float)

        # Generate data
        f = h5py.File(list_IDs_temp[0][0], 'r')
        for i, ID in enumerate(list_IDs_temp):
            # open right file if needed
            if f.filename != ID[0]:
                f.close()
                f = h5py.File(ID[0], 'r')
            X[i] = f['inputs'][ID[1]]#np.reshape(f['inputs'][ID[1]], (*self.indim, self.n_channels_in))
            intermediate_y[i] = f['intermediate_outputs'][ID[1]]#np.reshape(f['intermediate_outputs'][ID[1]], (*self.indim, self.n_channels_in))
            y[i] = f['outputs'][ID[1]]#np.reshape(f['outputs'][ID[1]], (*self.outdim, self.n_channels_out))
        f.close()
        return X, [intermediate_y,y]