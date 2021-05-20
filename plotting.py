import os
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import json
import pickle
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from NN.architectures import nrmse_2D_L2, norm_abs, unnorm_abs
from NN.architectures_cplx import ifft_layer, reconGAN_Wnet_intermediate
from utils.fastMRI_utils import ifft, fft
from tensorflow.keras.models import model_from_json
from matplotlib.colors import LogNorm


def abs_and_phase(x, y=None):
    if y is not None:
        compl = 1j*y + x
        return np.abs(compl), np.angle(compl)
    return np.abs(x), np.angle(x)

def real_and_imag(x):
    return np.real(x), np.imag(x)

def ifft2(real,imag):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(1j*imag+real)))

def simple_plot(cplximag):
    fig, axs = plt.subplots(2,1, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axs = axs.ravel()
    axs[0].imshow(np.abs(cplximag), cmap='gray')
    axs[1].imshow(np.angle(cplximag), cmap='gray')
    plt.show()

def hf_format(a,b, is_realim, output_ri):
    """ puts img or kspace into human friendly format"""
    #correct for zero centering
    if is_realim:
        if not output_ri:
            compl = a + 1j*b
            a = np.abs(compl)
            b = np.angle(compl)
    elif output_ri:
        m = 0.5*(a+1)
        p = b
        a = m * np.cos(p)
        b = m * np.sin(p)
    return a,b

def ifft_hf(a,b, is_realim, output_ri):
    a,b = hf_format(a,b, is_realim, True)
    compl = a + 1j*b
    imgcompl = ifft(compl)
    if output_ri:
        return np.real(imgcompl), np.imag(imgcompl)
    else:
        return np.abs(imgcompl), np.angle(imgcompl)

def fft_hf(a,b, is_realim, output_ri):
    a,b = hf_format(a,b, is_realim, True)
    compl = a + 1j*b
    imgcompl = np.fft.fft2(compl)#fft(compl)
    if output_ri:
        return np.real(imgcompl), np.imag(imgcompl)
    else:
        return np.abs(imgcompl), np.angle(imgcompl)

def model_output_plotting(data_files,model=None,model_path=None,plot=True, rewrite=False,only_best=True,
display_real_imag=True, input_image=False, ifft_output=False, ir_kspace=False, ir_img=False, intermediate_output=False, mask_kspace=False,
extra_plots=[],title=None,evaluation=True,reduced_mask_input=False):

    if model_path is not None:
        model_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if (
            os.path.isfile(os.path.join(model_path, f)) and ('.h5' in f))]
        if not rewrite:
            model_files = [f for f in model_files if not os.path.isfile(f[:-2]+'png')]
        if only_best:
            model_files = [f for f in model_files if 'best' in f]
    else:
        model_files = ['bla'] #to loop only once TODO find a better workaround

    for f in data_files:
        h5f = h5py.File(f, 'r')

        for model_file in model_files:
            if model_path is not None:
                model.load_weights(model_file)

            for i in [7]:#range(h5f['tests'].shape[1]):
                
                #input image
                x_r, x_i = hf_format(h5f['image_masked'][i,:,:,0], h5f['image_masked'][i,:,:,1],
                is_realim=ir_img,output_ri=display_real_imag)

                #ground truth
                Y_r, Y_i = hf_format(h5f['image_ground_truth'][i,:,:,0], h5f['image_ground_truth'][i,:,:,1],
                is_realim=ir_img,output_ri=display_real_imag)

                #evaluation of model if given one
                if input_image:
                    input_type = 'image_masked'
                else:
                    input_type = 'kspace_masked'
                # output_model = model.predict(np.expand_dims(h5f[input_type][i,:,:,:],axis=0))
                if mask_kspace:
                    if reduced_mask_input:
                        output_model = model.predict([np.expand_dims(np.expand_dims(h5f[input_type][i,:,:,:],axis=0),axis=4),
                        np.expand_dims(h5f['inverse_mask'][i,:,:],axis=0),
                        np.expand_dims(np.expand_dims(h5f[input_type+'_reduced'][i,:,:,:],axis=0),axis=4)])
                        # import pdb;pdb.set_trace()
                    else:
                        output_model = model.predict([np.expand_dims(np.expand_dims(h5f[input_type][i,:,:,:],axis=0),axis=4),np.expand_dims(h5f['inverse_mask'][i,:,:],axis=0)])
                else:
                    output_model = model.predict(np.expand_dims(np.expand_dims(h5f[input_type][i,:,:,:],axis=0),axis=4))
                if intermediate_output:
                    output_model = [np.reshape(output_model[0],(1,256,256,2)),np.reshape(output_model[1],(1,256,256,2))]
                else:
                    output_model = np.reshape(output_model,(1,256,256,2))
                # output_model = [np.expand_dims(h5f['kspace_masked'][i],axis=0),np.expand_dims(h5f['image_ground_truth'][i],axis=0)]

                #intermediate output
                if intermediate_output:
                    if intermediate_output=='kspace':
                        y_ks_r, y_ks_i = ifft_hf(output_model[0][0,:,:,0],output_model[0][0,:,:,1], is_realim=ir_kspace,output_ri=display_real_imag)
                    elif intermediate_output=='image':
                        y_ks_r, y_ks_i = hf_format(output_model[0][0,:,:,0],output_model[0][0,:,:,1], is_realim=ir_img,output_ri=display_real_imag)
                    else:
                        raise ValueError('intermediate_output should describe in which space it is given, maybe wrongly spelled?')
                    a = output_model[1][0,:,:,0]
                    b = output_model[1][0,:,:,1]
                else:
                    a = output_model[0,:,:,0]
                    b = output_model[0,:,:,1]
                    y_ks_r = np.zeros(a.shape)
                    y_ks_i = np.zeros(b.shape)

                #output
                if ifft_output:
                    y_img_r, y_img_i = ifft_hf(a,b, is_realim=ir_kspace,output_ri=display_real_imag)
                else:
                    y_img_r, y_img_i = hf_format(a,b, is_realim=ir_img,output_ri=display_real_imag)

                atag = 'real' if display_real_imag else 'abs'
                btag = 'imag' if display_real_imag else 'phase'

                display_dict = {
                    'Input (IMG)' : [x_r,x_i],
                    'Output (IMG)' : [y_img_r,y_img_i],
                    'GT (IMG)' : [Y_r,Y_i]
                }

                if intermediate_output:
                    display_dict['Interm Output (IMG)'] = [y_ks_r, y_ks_i]

                # extra plots # extra plots implemented : kspaces, kspaces_diff, images_diff
                for plot_name in extra_plots:
                    if plot_name=='kspaces':
                        display_dict['Input (KSPACE)'] = [h5f['kspace_masked'][i,:,:,0],h5f['kspace_masked'][i,:,:,1]]
                        if ifft_output:
                            display_dict['Output (KSPACE)'] = [output_model[0,:,:,0],output_model[0,:,:,1]]
                        if intermediate_output:
                            display_dict['Inter Output (KSPACE)'] = [output_model[0][0,:,:,0],output_model[0][0,:,:,1]]
                        display_dict['GT (KSPACE)'] = [h5f['kspace_ground_truth'][i,:,:,0],h5f['kspace_ground_truth'][i,:,:,1]]
                    elif plot_name=='kspaces_diff':
                        if intermediate_output:
                            display_dict['Interm_Output-Input (KSPACE)'] = [output_model[0][0,:,:,0]-h5f['kspace_masked'][i,:,:,0],
                                output_model[0][0,:,:,1]-h5f['kspace_masked'][i,:,:,1]]
                        if ifft_output:
                            display_dict['Input - Output (KSPACE)'] = [h5f['kspace_masked'][i,:,:,0] - output_model[0,:,:,0],
                            h5f['kspace_masked'][i,:,:,1] - output_model[0,:,:,1]]
                    elif plot_name=='mask':
                        display_dict['Mask (KSPACE)'] = [h5f['mask'][i,:,:],h5f['inverse_mask'][i,:,:]]
                    elif plot_name=='images_diff':
                        pass
                    else:
                        raise ValueError('Extra plot {} not implemented.'.format(plot_name))

                n_display = len(display_dict)
                fig, axs = plt.subplots(2,n_display, figsize=(100,100))
                if title:
                    if evaluation and os.path.isfile(os.path.join(model_path,'eval.json')):
                        with open(os.path.join(model_path,'eval.json')) as f:
                            json_eval = json.load(f)
                        evaluation_string = ' '.join([':'.join([mname,value]) for mname, value in json_eval.items()])
                        title = '\n'.join([title,evaluation_string])
                    fig.suptitle(title)
                axs = axs.ravel()
                i = 0
                for name, arr in display_dict.items():
                    if 'Mask' in name:
                        im1 = axs[i].matshow(arr[0],cmap='gray')
                        im2 = axs[i+n_display].matshow(arr[1],cmap='gray')
                    elif 'KSPACE' in name:
                        compl_arr = arr[0] + 1j* arr[1]
                        im1 = axs[i].matshow(np.abs(compl_arr)+0.01,cmap='gray',norm=LogNorm(vmin=0.01, vmax=10000))
                        im2 = axs[i+n_display].matshow(np.angle(compl_arr),cmap='gray')
                    else:
                        im1 = axs[i].matshow(arr[0],cmap='gray')
                        im2 = axs[i+n_display].matshow(arr[1],cmap='gray')
                    axs[i].title.set_text(' '.join([name,atag]))
                    axs[i+n_display].title.set_text(' '.join([name,btag]))
                    fig.colorbar(im1, ax=axs[i])
                    fig.colorbar(im2, ax=axs[i+n_display])
                    i+=1

                if plot:
                    plt.show()
                else:
                    plt.savefig(model_file[:-2]+'png')
                plt.close()

                # if display_kspaces:
                #     display_dict = {}
                #     ksa, ksb = hf_format(h5f['kspace_masked'][i,:,:,0], h5f['kspace_masked'][i,:,:,1],is_realim=ir_kspace,output_ri=display_real_imag)
                #     display_dict['Input (KS)'] = [ksa+2e-6, ksb+2e-6]
                #     if ifft_output:
                #         ksoa, ksob = hf_format(output_model[0,:,:,0],output_model[0,:,:,1],is_realim=ir_img, output_ri=display_real_imag)
                #     else:
                #         ksoa, ksob = fft_hf(output_model[0,:,:,0],output_model[0,:,:,1],is_realim=ir_img, output_ri=display_real_imag)
                #     display_dict['Output (KS)'] = [ksoa+2e-6, ksob+2e-6]
                #     ksgta, ksgtb = hf_format(h5f['kspace_ground_truth'][i,:,:,0], h5f['kspace_ground_truth'][i,:,:,1],is_realim=ir_kspace,output_ri=display_real_imag)
                #     display_dict['GT (KS)'] = [ksgta+2e-6,ksgtb+2e-6]
                #     display_dict['Output - input (KS)'] = [np.abs(ksoa-ksa)+2e-6,np.abs(ksob-ksb)+2e-6]
                #     n_display = len(display_dict)
                #     fig, axs = plt.subplots(2,n_display, figsize=(100,100))
                #     axs = axs.ravel()
                #     i = 0
                #     for name, arr in display_dict.items():
                #         im1 = axs[i].matshow(arr[0],cmap='gray',norm=LogNorm(vmin=1e-6,vmax=2), interpolation='none')
                #         im2 = axs[i+n_display].matshow(arr[1],cmap='gray',norm=LogNorm(vmin=1e-6,vmax=2), interpolation='none')
                #         axs[i].title.set_text(' '.join([name,atag]))
                #         axs[i+n_display].title.set_text(' '.join([name,btag]))
                #         fig.colorbar(im1, ax=axs[i])
                #         fig.colorbar(im2, ax=axs[i+n_display])
                #         i+=1
                #     if plot:
                #         plt.show()
                #     else:
                #         plt.savefig(model_file[:-3]+'ks'+'.png')
                #     plt.close()
        h5f.close()


def data_plot_slice(path,slice_number,plot=True,save_path=None):
    f = h5py.File(path, 'r')
    kspace = f['kspace_ground_truth'][slice_number,:,:,:]
    kspace_masked = f['kspace_masked'][slice_number,:,:,:]
    image = f['image_ground_truth'][slice_number,:,:,:]
    image_masked = f['image_masked'][slice_number,:,:,:]
    fig, axs = plt.subplots(2,4, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axs = axs.ravel()
    axs[0].imshow(kspace[:,:,0],cmap="gray")
    axs[0].title.set_text('kspace abs')
    axs[4].imshow(kspace[:,:,1],cmap="gray")
    axs[4].title.set_text('kspace phase')
    axs[1].imshow(kspace_masked[:,:,0],cmap="gray")
    axs[1].title.set_text('masked kspace abs')
    axs[5].imshow(kspace_masked[:,:,1],cmap="gray")
    axs[5].title.set_text('masked kspace phase')
    axs[2].imshow(image[:,:,0],cmap="gray")
    axs[2].title.set_text('image abs')
    axs[6].imshow(image[:,:,1],cmap="gray")
    axs[6].title.set_text('image phase')
    axs[3].imshow(image_masked[:,:,0],cmap="gray")
    axs[3].title.set_text('masked image abs')
    axs[7].imshow(image_masked[:,:,1],cmap="gray")
    axs[7].title.set_text('masked image phase')
    if save_path:
        plt.savefig(save_path)
    if plot:
        plt.show()
    plt.close()
    f.close()

def data_plot(imgs,plot=True,save_path=None):#,clipimage=-1):
    n_img = len(imgs)
    fig, axs = plt.subplots(2,n_img,figsize=(50,50))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axs = axs.ravel()
    vminr = np.amax(imgs[2][:,:,0])
    vmaxr = np.amin(imgs[2][:,:,0])
    vmini = np.amax(imgs[2][:,:,1])
    vmaxi = np.amin(imgs[2][:,:,1])
    for i,img in enumerate(imgs):
        axs[i].imshow(img[:,:,0],cmap='gray',vmin=vminr,vmax=vmaxr)
        axs[i+n_img].imshow(img[:,:,1],cmap='gray',vmin=vmini,vmax=vmaxi)
    if save_path:
        plt.savefig(save_path)
    if plot:
        plt.show()
    plt.close()


if __name__ == '__main__':
    tf.config.set_visible_devices([],'GPU')
    model_path = r'D:\NN_DATA\singlecoil_acc15_ksri_imgri_10midslices_densedpointmasked_point_mask\trainingsaves_ReconGAN_Unet_kspace_to_img_intermoutput_complex_Apr_29_08_32'
    param_file = os.path.join(model_path,'params_save.pck')
    with open(param_file,'rb') as pf:
       params =  pickle.load(pf)
    
    json_file = os.path.join(params['train_dir'],'model_save.json')
    if os.path.isfile(json_file):
        json_file = open(json_file, 'r')
        model = model_from_json(json_file.read())
        json_file.close()

    model.summary()
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    data_file_path = '\\'.join(model_path.split('\\')[:-1]+['val','file1000000.h5'])

    # plotting options
    # extra plots implemented : kspaces, kspaces_diff, images_diff
    extra_plots = ['kspaces','kspaces_diff','mask']

    display_real_imag = True
    display_kspaces=False
    only_best=True
    plot=False
    rewrite=True

    if os.path.isfile(data_file_path) and ('.h5' in data_file_path):
        data_files = [data_file_path]
    else:
        data_files = [os.path.join(data_file_path, f) for f in os.listdir(data_file_path) if (
                os.path.isfile(os.path.join(data_file_path, f)) and ('epoch' in f))]

    name = params['name'] if 'name' in params else 'TBD'
    mask_kspace = params['mask_kspace'] if 'mask_kspace' in params else None

    model_output_plotting(data_files,model,model_path,plot=plot,rewrite=rewrite, only_best=only_best, 
        intermediate_output=params['intermediate_output'], display_real_imag=display_real_imag, input_image=not params['input_kspace'], 
        ifft_output=not params['output_image'], ir_kspace=params['realimag_kspace'], ir_img=params['realimag_img'], mask_kspace=mask_kspace,
        extra_plots=extra_plots,title=name,reduced_mask_input=False)