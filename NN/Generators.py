import os
import numpy as np
import json
import h5py
from numpy.core.function_base import _needs_add_docstring
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """
    Versatile class for data generation in the IRM NN project.
    """

    def __init__(self, datadir_path, data_shape, input_kspace=True, output_image=True, intermediate_output=False, batch_size=8, shuffle=True, mask=False,ncoil=1):
        self.batch_size = batch_size
        self.input_kspace=input_kspace
        self.output_image=output_image
        self.intermediate_output=intermediate_output
        self.mask = mask
        self.data_shape=data_shape
        self.list_IDs = []
        self.n_coil = ncoil
        with open(os.path.join(datadir_path,'index.json'), 'r') as fp:
            index_dict = json.load(fp)
            for fname in index_dict:
                for i in range(index_dict[fname]):
                    self.list_IDs.append([fname,i])
            fp.close()
        # if fraction:
        #     self.list_IDs = self.list_IDs[:int(fraction*len(self.list_IDs))]
        # with open(os.path.join(datadir_path,'format.json'), 'r') as fp:
        #     self.data_shape = json.load(fp)
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

    def __data_generation(self, list_IDs_temp):
        #TODO create the arrays as belonging to the class, to not have to recreate one on each data_generation, just fill it with values
        X = np.empty((self.batch_size, self.n_coil, *self.data_shape))
        #  input_kspace=True, output_image=True, intermediate_output=False
        if self.intermediate_output:
            intermediate_y = np.empty((self.batch_size, *self.data_shape),dtype=np.float32)
        if self.mask:
            mask = np.empty((self.batch_size, 256,256),dtype=np.float32)
        y = np.empty((self.batch_size, *self.data_shape),dtype=np.float32)

        inc = 'kspace' if self.input_kspace else 'image'
        outg = 'image' if self.output_image else 'kspace'

        # Generate data
        f = h5py.File(list_IDs_temp[0][0], 'r')
        for i, ID in enumerate(list_IDs_temp):
            # open right file if needed
            if f.filename != ID[0]:
                f.close()
                f = h5py.File(ID[0], 'r')
            
            X[i] = np.reshape(f['{}_masked'.format(inc)][ID[1]], (self.n_coil,*self.data_shape))
            if self.intermediate_output=='kspace':
                intermediate_y[i] = np.reshape(f['{}_ground_truth'.format(self.intermediate_output)][ID[1]], (self.n_coil,*self.data_shape))
            elif self.intermediate_output=='image':
                intermediate_y[i] = np.reshape(f['{}_ground_truth'.format(self.intermediate_output)][ID[1]], self.data_shape)
            if self.mask:
                mask[i] = np.reshape(f['inverse_mask'][ID[1]], (256,256))
            y[i] = np.reshape(f['{}_ground_truth'.format(outg)][ID[1]], self.data_shape)
        f.close()
        if self.mask:
            X = [X,mask]
        if self.intermediate_output:
            y = [intermediate_y,y]
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

# class BaseDataGenerator(Sequence):
#     """
#     Base class for data generators, holds the basic common functions.
#     """
    
#     def __init__(self, datadir_path, batch_size=8, shuffle=True):
#         """
#         Initialization.
#         """
#         self.batch_size = batch_size
#         self.list_IDs = []
#         with open(os.path.join(datadir_path,'index.json'), 'r') as fp:
#             index_dict = json.load(fp)
#             for fname in index_dict:
#                 for i in range(index_dict[fname]):
#                     self.list_IDs.append([fname,i])
#             fp.close()
#         with open(os.path.join(datadir_path,'format.json'), 'r') as fp:
#             self.data_shape = json.load(fp)
#         self.shuffle = shuffle
#         self.on_epoch_end()

#     def __len__(self):
#         """
#         Denotes the number of batches per epoch.
#         """
#         return int(np.floor(len(self.list_IDs) / self.batch_size))

#     def __getitem__(self, index):
#         """
#         Generate one batch of data.
#         """
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

#     def on_epoch_end(self):
#         """
#         Updates indexes after each epoch.
#         """
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)


# class DataGenerator_kspace_img(BaseDataGenerator):
#     """
#     Generates data for a Wnet_like architecture.
#     """
#     def __data_generation(self, list_IDs_temp):
#         """
#         Generates data containing batch_size samples.
#         """
#         X = np.empty((self.batch_size, *self.data_shape))
#         intermediate_y = np.empty((self.batch_size, *self.data_shape))
#         y = np.empty((self.batch_size, *self.data_shape))

#         # Generate data
#         f = h5py.File(list_IDs_temp[0][0], 'r')
#         for i, ID in enumerate(list_IDs_temp):
#             # open right file if needed
#             if f.filename != ID[0]:
#                 f.close()
#                 f = h5py.File(ID[0], 'r')
#             X[i] = f['kspace_masked'][ID[1]]
#             intermediate_y[i] = f['image_ground_truth'][ID[1]]
#             y[i] = f['image_ground_truth'][ID[1]]
#         f.close()
#         return X, [intermediate_y,y]

#     def __getitem__(self, index):
#         """
#         Generate one batch of data.
#         """
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

# class DataGenerator_kspace_img_interm_kspace(BaseDataGenerator):
#     """
#     Generates data for a Wnet_like architecture.
#     """
#     def __data_generation(self, list_IDs_temp):
#         """
#         Generates data containing batch_size samples.
#         """
#         X = np.empty((self.batch_size, *self.data_shape))
#         intermediate_y = np.empty((self.batch_size, *self.data_shape))
#         y = np.empty((self.batch_size, *self.data_shape))

#         # Generate data
#         f = h5py.File(list_IDs_temp[0][0], 'r')
#         for i, ID in enumerate(list_IDs_temp):
#             # open right file if needed
#             if f.filename != ID[0]:
#                 f.close()
#                 f = h5py.File(ID[0], 'r')
#             X[i] = f['kspace_masked'][ID[1]]
#             intermediate_y[i] = f['kspace_ground_truth'][ID[1]]
#             y[i] = f['image_ground_truth'][ID[1]]
#         f.close()
#         return X, [intermediate_y,y]

#     def __getitem__(self, index):
#         """
#         Generate one batch of data.
#         """
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

# class DataGenerator_kspace_img_interm_kspace_onlyabsimg(BaseDataGenerator):
#     """
#     Generates data for a Wnet_like architecture.
#     """
#     def __data_generation(self, list_IDs_temp):
#         """
#         Generates data containing batch_size samples.
#         """
#         X = np.empty((self.batch_size, *self.data_shape))
#         intermediate_y = np.empty((self.batch_size, *self.data_shape))
#         y = np.empty((self.batch_size, *self.data_shape[:2],1))

#         # Generate data
#         f = h5py.File(list_IDs_temp[0][0], 'r')
#         for i, ID in enumerate(list_IDs_temp):
#             # open right file if needed
#             if f.filename != ID[0]:
#                 f.close()
#                 f = h5py.File(ID[0], 'r')
#             X[i] = f['kspace_masked'][ID[1]]
#             intermediate_y[i] = f['kspace_ground_truth'][ID[1]]
#             y[i] = np.expand_dims(np.abs(f['image_ground_truth'][ID[1]][:,:,0]+1j*f['image_ground_truth'][ID[1]][:,:,1]),axis=-1)
#         f.close()
#         return X, [intermediate_y,y]

#     def __getitem__(self, index):
#         """
#         Generate one batch of data.
#         """
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

# class DataGenerator_kspace_to_img(BaseDataGenerator):
#     """
#     Generates data for a Wnet_like architecture.
#     """
#     def __data_generation(self, list_IDs_temp):
#         """
#         Generates data containing batch_size samples.
#         """
#         X = np.empty((self.batch_size, *self.data_shape))
#         y = np.empty((self.batch_size, *self.data_shape))

#         # Generate data
#         f = h5py.File(list_IDs_temp[0][0], 'r')
#         for i, ID in enumerate(list_IDs_temp):
#             # open right file if needed
#             if f.filename != ID[0]:
#                 f.close()
#                 f = h5py.File(ID[0], 'r')
#             X[i] = f['kspace_masked'][ID[1]]
#             y[i] = f['image_ground_truth'][ID[1]]
#         f.close()
#         return X, y

#     def __getitem__(self, index):
#         """
#         Generate one batch of data.
#         """
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

# class DataGenerator_img(BaseDataGenerator):
#     """
#     Generates data for a Wnet_like architecture.
#     """
#     def __data_generation(self, list_IDs_temp):
#         """
#         Generates data containing batch_size samples.
#         """
#         X = np.empty((self.batch_size, *self.data_shape),dtype=np.float32)
#         y = np.empty((self.batch_size, *self.data_shape),dtype=np.float32)

#         # Generate data
#         f = h5py.File(list_IDs_temp[0][0], 'r')
#         for i, ID in enumerate(list_IDs_temp):
#             # open right file if needed
#             if f.filename != ID[0]:
#                 f.close()
#                 f = h5py.File(ID[0], 'r')
#             X[i] = f['image_masked'][ID[1]]
#             y[i] = f['image_ground_truth'][ID[1]]
#         f.close()
#         return X,y

#     def __getitem__(self, index):
#         """
#         Generate one batch of data.
#         """
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

# class DataGenerator_img_abs(BaseDataGenerator):
#     """
#     Generates data for a Wnet_like architecture.
#     """
#     def __data_generation(self, list_IDs_temp):
#         """
#         Generates data containing batch_size samples.
#         """
#         X = np.empty((self.batch_size, 256,256,1),dtype=np.float32)
#         y = np.empty((self.batch_size, 256,256,1),dtype=np.float32)

#         # Generate data
#         f = h5py.File(list_IDs_temp[0][0], 'r')
#         for i, ID in enumerate(list_IDs_temp):
#             # open right file if needed
#             if f.filename != ID[0]:
#                 f.close()
#                 f = h5py.File(ID[0], 'r')
#             X[i] = np.reshape(f['image_masked'][ID[1],:,:,0],(256,256,1))
#             y[i] = np.reshape(f['image_ground_truth'][ID[1],:,:,0],(256,256,1))
#         f.close()
#         return X,y

#     def __getitem__(self, index):
#         """
#         Generate one batch of data.
#         """
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y


# class DataGenerator_fullimg_abs_interm(BaseDataGenerator):
#     """
#     Generates data for a Wnet_like architecture.
#     """
#     def __data_generation(self, list_IDs_temp):
#         """
#         Generates data containing batch_size samples.
#         """
#         X = np.empty((self.batch_size, 256,256,2),dtype=np.float32)
#         y_interm = np.empty((self.batch_size, 256,256,2),dtype=np.float32)
#         y = np.empty((self.batch_size, 256,256,1),dtype=np.float32)


#         # Generate data
#         f = h5py.File(list_IDs_temp[0][0], 'r')
#         for i, ID in enumerate(list_IDs_temp):
#             # open right file if needed
#             if f.filename != ID[0]:
#                 f.close()
#                 f = h5py.File(ID[0], 'r')
#             X[i] = np.reshape(f['kspace_masked'][ID[1],:,:,:],(256,256,2))
#             y_interm[i] = np.reshape(f['kspace_ground_truth'][ID[1],:,:,:],(256,256,2))
#             img = f['image_ground_truth'][ID[1],:,:,0] + 1j*f['image_ground_truth'][ID[1],:,:,1]
#             y[i] = np.reshape(np.abs(img),(256,256,1))
#         f.close()
#         return X,[y_interm,y]

#     def __getitem__(self, index):
#         """
#         Generate one batch of data.
#         """
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y


# class DataGenerator_kspace(BaseDataGenerator):
#     """
#     Generates data for a Wnet_like architecture.
#     """
#     def __data_generation(self, list_IDs_temp):
#         """
#         Generates data containing batch_size samples.
#         """
#         X = np.empty((self.batch_size, *self.data_shape))
#         y = np.empty((self.batch_size, *self.data_shape))

#         # Generate data
#         f = h5py.File(list_IDs_temp[0][0], 'r')
#         for i, ID in enumerate(list_IDs_temp):
#             # open right file if needed
#             if f.filename != ID[0]:
#                 f.close()
#                 f = h5py.File(ID[0], 'r')
#             X[i] = f['kspace_masked'][ID[1]]
#             y[i] = f['kspace_ground_truth'][ID[1]]
#         f.close()
#         return X,y
        
#     def __getitem__(self, index):
#         """
#         Generate one batch of data.
#         """
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y


# class DataGenerator_complex(BaseDataGenerator):
#     """
#     Generates data for a Wnet_like architecture.
#     """
#     def __data_generation(self, list_IDs_temp):
#         """
#         Generates data containing batch_size samples.
#         """
#         X = np.empty((self.batch_size, *self.data_shape,1))
#         y_int = np.empty((self.batch_size, *self.data_shape,1))
#         y = np.empty((self.batch_size, *self.data_shape,1))

#         # Generate data
#         f = h5py.File(list_IDs_temp[0][0], 'r')
#         for i, ID in enumerate(list_IDs_temp):
#             # open right file if needed
#             if f.filename != ID[0]:
#                 f.close()
#                 f = h5py.File(ID[0], 'r')
#             X[i] = np.expand_dims(f['kspace_masked'][ID[1]],axis=-1)
#             y_int[i] = np.expand_dims(f['kspace_ground_truth'][ID[1]],axis=-1)
#             y[i] = np.expand_dims(f['image_ground_truth'][ID[1]],axis=-1)
#         f.close()
#         return X,[y_int,y]
        
#     def __getitem__(self, index):
#         """
#         Generate one batch of data.
#         """
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

# class DataGenerator_complex_image(BaseDataGenerator):
#     """
#     Generates data for a Wnet_like architecture.
#     """
#     def __data_generation(self, list_IDs_temp):
#         """
#         Generates data containing batch_size samples.
#         """
#         X = np.empty((self.batch_size, *self.data_shape,1))
#         y = np.empty((self.batch_size, *self.data_shape,1))

#         # Generate data
#         f = h5py.File(list_IDs_temp[0][0], 'r')
#         for i, ID in enumerate(list_IDs_temp):
#             # open right file if needed
#             if f.filename != ID[0]:
#                 f.close()
#                 f = h5py.File(ID[0], 'r')
#             X[i] = np.expand_dims(f['image_masked'][ID[1]],axis=-1)
#             y[i] = np.expand_dims(f['image_ground_truth'][ID[1]],axis=-1)
#         f.close()
#         return X,y
        
#     def __getitem__(self, index):
#         """
#         Generate one batch of data.
#         """
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

# generator_dict = {
#     'imgtoimg':DataGenerator_img,
#     'kspacetokspace':DataGenerator_kspace,
#     'kspacetoimg':DataGenerator_kspace_to_img,
#     'kspacetokspace&img':DataGenerator_kspace_img
# }

# def find_generator(input_kspace, output_image, intermediate_output=False, is_complex = False):
#     if is_complex:
#         if input_kspace:
#             return DataGenerator_complex
#         else:
#             return DataGenerator_complex_image
#     else:
#         if input_kspace:
#             if output_image:
#                 if intermediate_output=='image':
#                     return DataGenerator_kspace_img
#                 elif intermediate_output=='kspace':
#                     return DataGenerator_kspace_img_interm_kspace
#                 else:
#                     return DataGenerator_kspace_to_img
#             else:
#                 return DataGenerator_kspace
#         else:
#             return DataGenerator_img