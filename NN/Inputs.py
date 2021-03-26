import os
import numpy as np
import h5py
import json
from utils.fastMRI_utils import image_from_kspace, crop, fft, ifft
from progress.bar import IncrementalBar
# import matplotlib.pyplot as plt

# TODO use mongoDB to handle file locations

def prepare_loic_dataset():
    pass

def prepare_datasets(datapath, workdirpath, dataset_type, 
    image_shape=(256,256),
    input_mask=None,
    fraction=None, n_slice_per_file=None,
    absphase_img=False,absphase_kspace=False,
    normalise_image=True, center_normalised_values=True):
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

    if files != []:
        bar = IncrementalBar('{} dataset_files'.format(dataset_type), max = len(files))

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
        for i, kdata_raw in enumerate(h5f['kspace']):
            if i<imin or i>=imax:
                continue

            ### cropping
            image_clean = np.fft.fftshift(np.fft.ifft2(kdata_raw))
            image_clean = crop(image_clean,size=image_shape)
            kdata_clean = np.fft.fft2(image_clean)

            ### normalise kspace
            kdata_clean_abs = np.abs(kdata_clean)
            kdata_clean_angle = np.angle(kdata_clean)
            kdata_clean_abs /= np.amax(kdata_clean_abs)
            kdata_clean = kdata_clean_abs * (np.cos(kdata_clean_angle) + 1j*np.sin(kdata_clean_angle))

            ### apply mask
            if input_mask:
                kdata = input_mask(kdata_clean)
            else:
                kdata = kdata_clean
            import pdb;pdb.set_trace()

            image = np.fft.ifft2(kdata)
            image_clean = np.fft.ifft2(kdata_clean)
            ### normalisation + centering values around zero
            if normalise_image:

                image_clean_abs = np.abs(image_clean)
                image_clean_angle = np.angle(image_clean)
                image_clean_abs = image_clean_abs/np.amax(image_clean_abs)
                image_clean = image_clean_abs * (np.cos(image_clean_angle) + 1j*np.sin(image_clean_angle))

                image_abs = np.abs(image)
                image_angle = np.angle(image)
                image_abs = image_abs/np.amax(image_abs)
                image = image_abs * (np.cos(image_angle) + 1j*np.sin(image_angle))


            # finally filling the tables
            if center_normalised_values:
                if absphase_kspace:
                    kdata_array[k,:,:,0] = 2*np.abs(kdata)-1
                    kdata_array[k,:,:,1] = np.angle(kdata)
                    kdata_clean_array[k,:,:,0] = 2*np.abs(kdata_clean)-1
                    kdata_clean_array[k,:,:,1] = np.angle(kdata_clean)
                else:
                    kdata_array[k,:,:,0] = np.real(kdata)
                    kdata_array[k,:,:,1] = np.imag(kdata)
                    kdata_clean_array[k,:,:,0] = np.real(kdata_clean)
                    kdata_clean_array[k,:,:,1] = np.imag(kdata_clean)


                if absphase_img:
                    image_array[k,:,:,0] = 2*np.abs(image)-1
                    image_array[k,:,:,1] = np.angle(image)
                    image_clean_array[k,:,:,0] = 2*np.abs(image_clean)-1
                    image_clean_array[k,:,:,1] = np.angle(image_clean)
                else:
                    image_array[k,:,:,0] = np.real(image)
                    image_array[k,:,:,1] = np.imag(image)
                    image_clean_array[k,:,:,0] = np.real(image_clean)
                    image_clean_array[k,:,:,1] = np.imag(image_clean)
            else:
                if absphase_kspace:
                    kdata_array[k,:,:,0] = np.abs(kdata)
                    kdata_array[k,:,:,1] = np.angle(kdata)
                    kdata_clean_array[k,:,:,0] = np.abs(kdata_clean)
                    kdata_clean_array[k,:,:,1] = np.angle(kdata_clean)
                else:
                    kdata_array[k,:,:,0] = np.real(kdata)
                    kdata_array[k,:,:,1] = np.imag(kdata)
                    kdata_clean_array[k,:,:,0] = np.real(kdata_clean)
                    kdata_clean_array[k,:,:,1] = np.imag(kdata_clean)


                if absphase_img:
                    image_array[k,:,:,0] = np.abs(image)
                    image_array[k,:,:,1] = np.angle(image)
                    image_clean_array[k,:,:,0] = np.abs(image_clean)
                    image_clean_array[k,:,:,1] = np.angle(image_clean)
                else:
                    image_array[k,:,:,0] = np.real(image)
                    image_array[k,:,:,1] = np.imag(image)
                    image_clean_array[k,:,:,0] = np.real(image_clean)
                    image_clean_array[k,:,:,1] = np.imag(image_clean)
            import pdb;pdb.set_trace()
            k+=1
        h5f.close()
        outfile = h5py.File(filepath,'w')
        outfile.create_dataset('kspace_masked', data=kdata_array)
        outfile.create_dataset('kspace_ground_truth', data=kdata_clean_array)
        outfile.create_dataset('image_masked', data=image_array)
        outfile.create_dataset('image_ground_truth', data=image_clean_array)
        outfile.close()
        bar.next()
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
