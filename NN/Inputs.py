import os
import numpy as np
import h5py
import json
from utils.fastMRI_utils import image_from_kspace, crop, fft, ifft
# import matplotlib.pyplot as plt

# TODO use mongoDB to handle file locations

def prepare_image(kdata_raw,crop_size=(320,320),mask=None):

    image_clean = ifft(kdata_raw)

    if crop:
        image_clean = crop(image_clean,size=crop_size)
    
    kdata_clean = fft(image_clean)
    # imtest = ifft(kdata_clean)

    # fig, axs = plt.subplots(3,2, figsize=(20, 20))
    # fig.subplots_adjust(hspace=0.1, wspace=0.1)
    # axs = axs.ravel()
    # axs[0].imshow(np.abs(imtest))
    # axs[1].imshow(np.angle(imtest))
    # plt.show()

    kdata_clean_abs = np.abs(kdata_clean)
    kdata_clean_angle = np.angle(kdata_clean)
    kdata_clean_abs /= np.amax(kdata_clean_abs)

    kdata_clean = kdata_clean_abs * (np.cos(kdata_clean_angle) + 1j*np.sin(kdata_clean_angle))
    # imtest_2 = ifft(kdata_clean_2)

    # fig, axs = plt.subplots(2,2, figsize=(20, 20))
    # fig.subplots_adjust(hspace=0.1, wspace=0.1)
    # axs = axs.ravel()
    # axs[0].imshow(np.abs(imtest))
    # axs[1].imshow(np.angle(imtest))
    # axs[2].imshow(np.abs(imtest_2))
    # axs[3].imshow(np.angle(imtest_2))
    # plt.show()
    # import pdb;pdb.set_trace()
    if mask:
        kdata = mask(kdata_clean)
    else:
        kdata = kdata_clean

    image = ifft(kdata)
    
    return kdata, kdata_clean, image, image_clean
    

def prepare_datasets(datapath, workdirpath, dataset_type, 
    image_shape=(256,256),
    input_mask=None,
    fraction=None, n_slice_per_file=None):
    """
    Prepares the work directory, and prepares data into easy-to-used, eventually masked data.
    """
    # getting the list of usable files
    files = [f for f in os.listdir(datapath) if (
        os.path.isfile(os.path.join(datapath, f)) and ('.h5' in f))]
    if fraction:
        files = files[:int(np.floor(fraction*len(files)))]


    output_dir = os.path.join(workdirpath,dataset_type)
    if not os.path.isdir(output_dir):
         os.mkdir(output_dir)

    index_dict = {}

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
        index_dict[filepath] = n_slices

        kdata_array = np.empty((n_slices, *image_shape, 2))
        kdata_clean_array = np.empty((n_slices, *image_shape, 2))
        image_array = np.empty((n_slices, *image_shape, 2))
        image_clean_array = np.empty((n_slices, *image_shape, 2))

        k = 0
        imin = int(np.floor(h5f['kspace'].shape[0]/2 - n_slices/2))
        imax = imin + n_slices
        for i, slic in enumerate(h5f['kspace']):
            if i<imin or i>=imax:
                continue
            kdata, kdata_clean, image, image_clean = prepare_image(slic, crop_size=image_shape,mask=input_mask)
            kdata_array[k,:,:,0] = np.abs(kdata)
            kdata_array[k,:,:,1] = np.angle(kdata)

            kdata_clean_array[k,:,:,0] = np.abs(kdata_clean)
            kdata_clean_array[k,:,:,1] = np.angle(kdata_clean)

            image_array[k,:,:,0] = np.abs(image)
            image_array[k,:,:,1] = np.angle(image)

            image_clean_array[k,:,:,0] = np.abs(image_clean)
            image_clean_array[k,:,:,1] = np.angle(image_clean)
            k+=1
        h5f.close()
        outfile = h5py.File(filepath,'w')
        outfile.create_dataset('kspace_masked', data=kdata_array)
        outfile.create_dataset('kspace_ground_truth', data=kdata_clean_array)
        outfile.create_dataset('image_masked', data=image_array)
        outfile.create_dataset('image_ground_truth', data=image_clean_array)
        outfile.close()
    if index_dict!={}:
        with open(os.path.join(output_dir,'index.json'), 'w') as fp:
            json.dump(index_dict, fp)
        with open(os.path.join(output_dir,'format.json'), 'w') as fp:
            json.dump((*image_shape, 2) , fp)
    return output_dir

# def test_data(path,agent,batch_size=5):
#     """
#     Tests the agent on the data in the given path.
#     """
#     files = [f for f in os.listdir(path) if (
#         os.path.isfile(os.path.join(path, f)) and ('.h5' in f))]
        
#     for f in files:
#         h5f = h5py.File(os.path.join(path, f), 'r+')
#         inputs = np.reshape(h5f['inputs'],h5f['inputs'].shape)#np.reshape(h5f['inputs'],(h5f['inputs'].shape[0],*indim))
#         test_outputs = agent.test(inputs,batch_size=batch_size)
#         h5f.create_dataset('tests',data=test_outputs)
#         h5f.close()
