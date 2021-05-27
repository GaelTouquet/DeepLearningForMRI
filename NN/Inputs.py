import os
import numpy as np
import h5py
import json
from utils.fastMRI_utils import image_from_kspace, crop, fft, ifft
from progress.bar import IncrementalBar
import pydicom

# TODO use mongoDB to handle file locations

def prepare_datasets_fromdicom(datapath, workdirpath, dataset_type,
    image_shape=(256,256),fraction=None, n_slice_per_file=None, input_mask=None,
    realimag_img=True,realimag_kspace=True):

    files = [f for f in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, f))]
    if fraction:
        files = files[:int(np.floor(fraction*len(files)))]

    output_dir = os.path.join(workdirpath,dataset_type)

    index_dict = {}

    if files != []:
        bar = IncrementalBar('{} dataset_files'.format(dataset_type), max = len(files))

    for patient in files:
        filepath = os.path.join(output_dir,patient)
        if os.path.isfile(filepath):
            continue
        index_dict[filepath] = n_slice_per_file
        slicefiles = [os.path.join(datapath, patient, f) for f in os.listdir(os.path.join(datapath,patient)) if os.path.isfile(os.path.join(datapath, patient, f))]
        kdata_array = np.empty((n_slice_per_file, *image_shape, 2), np.float32)
        kdata_clean_array = np.empty((n_slice_per_file, *image_shape, 2), np.float32)
        image_array = np.empty((n_slice_per_file, *image_shape, 2), np.float32)
        image_clean_array = np.empty((n_slice_per_file, *image_shape, 2), np.float32)
        mask_array = np.empty((n_slice_per_file, *image_shape), np.float32)
        inverse_mask_array = np.empty((n_slice_per_file, *image_shape), np.float32)
    
        for k, slicef in enumerate(slicefiles):
            ds = pydicom.dcmread(slicef)
            image_clean = ds.pixel_array*float(ds['RescaleSlope']) + float(ds['RescaleIntercept'])
            kdata_clean = fft(image_clean)
            ### apply mask
            if input_mask:
                mask = input_mask.get_mask(kdata_clean)
                kdata = kdata_clean * mask + 0.0
                mask_array[k,:,:] = mask
                inverse_mask = mask==0
                inverse_mask = inverse_mask.astype(np.float)
                inverse_mask_array[k,:,:] = inverse_mask
            else:
                kdata = kdata_clean
            
            image = ifft(kdata)

            # filling arrays
            if realimag_kspace:
                kdata_array[k,:,:,0] = np.real(kdata)
                kdata_array[k,:,:,1] = np.imag(kdata)
                kdata_clean_array[k,:,:,0] = np.real(kdata_clean)
                kdata_clean_array[k,:,:,1] = np.imag(kdata_clean)
            else:
                kdata_array[k,:,:,0] = np.abs(kdata)
                kdata_array[k,:,:,1] = np.angle(kdata)
                kdata_clean_array[k,:,:,0] = np.abs(kdata_clean)
                kdata_clean_array[k,:,:,1] = np.angle(kdata_clean)

            if realimag_img:
                image_array[k,:,:,0] = np.real(image)
                image_array[k,:,:,1] = np.imag(image)
                image_clean_array[k,:,:,0] = np.real(image_clean)
                image_clean_array[k,:,:,1] = np.imag(image_clean)
            else:
                image_array[k,:,:,0] = np.abs(image)
                image_array[k,:,:,1] = np.angle(image)
                image_clean_array[k,:,:,0] = np.abs(image_clean)
                image_clean_array[k,:,:,1] = np.angle(image_clean)

        outfile = h5py.File(filepath,'w')
        outfile.create_dataset('kspace_masked', data=kdata_array)
        outfile.create_dataset('kspace_ground_truth', data=kdata_clean_array)
        outfile.create_dataset('image_masked', data=image_array)
        outfile.create_dataset('image_ground_truth', data=image_clean_array)
        outfile.create_dataset('mask', data=mask_array)
        outfile.create_dataset('inverse_mask', data=inverse_mask_array)
        outfile.close()
        bar.next()
    if index_dict!={}:
        with open(os.path.join(output_dir,'index.json'), 'w') as fp:
            json.dump(index_dict, fp)
        with open(os.path.join(output_dir,'format.json'), 'w') as fp:
            json.dump((*image_shape, 2) , fp)
    return output_dir


def prepare_datasets(datapath, workdirpath, dataset_type, 
    image_shape=(256,256),
    input_mask=None,
    fraction=None, n_slice_per_file=None,
    realimag_img=True,realimag_kspace=True,
    kspace_norm=None,img_norm=None,post_mask_shape=None, ncoil=1):
    """
    Prepares the work directory, and prepares data into easy-to-used, eventually masked data.
    """
    #TODO finish from_dicom implementation
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
        # if from_dicom:
        #     filepath = os.path.join(output_dir,f)
        #     if os.path.isfile(filepath):
        #         continue
        h5f = h5py.File(os.path.join(datapath, f), 'r')
        if ('kspace' not in h5f):
            continue

        if n_slice_per_file is None:
            n_slices = h5f['kspace'].shape[0]
        else:
            n_slices = n_slice_per_file
        index_dict[filepath] = n_slices
        n_slices = (n_slices,ncoil)

        if post_mask_shape:
            kdata_reduced_array = np.empty((*n_slices, *post_mask_shape, 2), np.float32)
        kdata_array = np.empty((*n_slices, *image_shape, 2), np.float32)
        kdata_clean_array = np.empty((*n_slices, *image_shape, 2), np.float32)
        image_array = np.empty((n_slices[0], *image_shape, 2), np.float32)
        image_clean_array = np.empty((n_slices[0], *image_shape, 2), np.float32)
        mask_array = np.empty((n_slices[0], *image_shape), np.float32)
        inverse_mask_array = np.empty((n_slices[0], *image_shape), np.float32)

        k = 0
        imin = int(np.floor(h5f['kspace'].shape[0]/2 - n_slices[0]/2))
        imax = imin + n_slices[0]
        for i, slice in enumerate(h5f['kspace']):
            if i<imin or i>=imax:
                continue
            
            ### apply mask
            if input_mask:
                if post_mask_shape:
                    kdata_reduced, kdata = input_mask(kdata_clean)
                    mask_array[k,:,:] = input_mask.full_mask(kdata_clean)
                    inverse_mask_array[k,:,:] = input_mask.inverse_full_mask(kdata_clean)
                else:
                    mask = input_mask.get_mask(None)
                    mask_array[k,:,:] = mask
                    inverse_mask = mask==0
                    inverse_mask = inverse_mask.astype(np.float32)
                    inverse_mask_array[k,:,:] = inverse_mask

            image_for_rss = np.zeros(image_shape, np.complex128)
            image_clean_for_rss = np.zeros(image_shape, np.complex128)
            for j, kdata_raw in enumerate(slice):
                if j >= ncoil:
                    print('Warning, not using all availeble coils!')
                    continue
                ### cropping
                image_clean = ifft(kdata_raw)
                image_clean = crop(image_clean,size=image_shape)

                #normalize image
                if img_norm['np']:
                    image_clean = img_norm['np'](image_clean)

                kdata_clean = fft(image_clean)

                if kspace_norm['np']:
                    kdata_clean = kspace_norm['np'](kdata_clean)

                ### apply mask
                if input_mask:
                    if post_mask_shape:
                        kdata_reduced, kdata = input_mask(kdata_clean)
                    else:
                        # mask = input_mask.get_mask(kdata_clean)
                        kdata = kdata_clean * mask + 0.0
                else:
                    kdata = kdata_clean

                image = ifft(kdata)
                image_clean = ifft(kdata_clean)

                # filling arrays
                if realimag_kspace:
                    kdata_array[k,j,:,:,0] = np.real(kdata)
                    kdata_array[k,j,:,:,1] = np.imag(kdata)
                    kdata_clean_array[k,j,:,:,0] = np.real(kdata_clean)
                    kdata_clean_array[k,j,:,:,1] = np.imag(kdata_clean)
                    if post_mask_shape:
                        kdata_reduced_array[k,j,:,:,0] = np.real(kdata_reduced)
                        kdata_reduced_array[k,j,:,:,1] = np.imag(kdata_reduced)
                else:
                    kdata_array[k,j,:,:,0] = np.abs(kdata)
                    kdata_array[k,j,:,:,1] = np.angle(kdata)
                    kdata_clean_array[k,j,:,:,0] = np.abs(kdata_clean)
                    kdata_clean_array[k,j,:,:,1] = np.angle(kdata_clean)
                    if post_mask_shape:
                        kdata_reduced_array[k,j,:,:,0] = np.abs(kdata_reduced)
                        kdata_reduced_array[k,j,:,:,1] = np.angle(kdata_reduced)
                image_for_rss += image * np.conjugate(image)
                image_clean_for_rss += image_clean * np.conjugate(image_clean)
            image_for_rss = np.sqrt(image_for_rss)
            image_for_rss = image_for_rss.astype(np.float32)
            image_clean_for_rss = np.sqrt(image_clean_for_rss)
            image_clean_for_rss = image_clean_for_rss.astype(np.float32)
            image = image_for_rss
            image_clean = image_clean_for_rss
            if realimag_img:
                image_array[k,:,:,0] = np.real(image)
                image_array[k,:,:,1] = np.imag(image)
                image_clean_array[k,:,:,0] = np.real(image_clean)
                image_clean_array[k,:,:,1] = np.imag(image_clean)
            else:
                image_array[k,:,:,0] = np.abs(image)
                image_array[k,:,:,1] = np.angle(image)
                image_clean_array[k,:,:,0] = np.abs(image_clean)
                image_clean_array[k,:,:,1] = np.angle(image_clean)

            k+=1
        h5f.close()
        outfile = h5py.File(filepath,'w')
        outfile.create_dataset('kspace_masked', data=kdata_array)
        if post_mask_shape:
            outfile.create_dataset('kspace_masked_reduced', data=kdata_reduced_array)
        outfile.create_dataset('kspace_ground_truth', data=kdata_clean_array)
        outfile.create_dataset('image_masked', data=image_array)
        outfile.create_dataset('image_ground_truth', data=image_clean_array)
        outfile.create_dataset('mask', data=mask_array)
        outfile.create_dataset('inverse_mask', data=inverse_mask_array)
        outfile.close()
        bar.next()
    if index_dict!={}:
        with open(os.path.join(output_dir,'index.json'), 'w') as fp:
            json.dump(index_dict, fp)
        with open(os.path.join(output_dir,'format.json'), 'w') as fp:
            json.dump((*image_shape, 2) , fp)
    return output_dir