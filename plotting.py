import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from NN.architectures import get_unet, get_unet_old

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

def plotting(data_files,show_abs_and_phase,show_kspaces,model=None,model_path=None,plot=True, rewrite=False):

    if model_path is not None:
        model_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if (
            os.path.isfile(os.path.join(model_path, f)) and ('.h5' in f))]
        if not rewrite:
            model_files = [f for f in model_files if not os.path.isfile(f[:-2]+'png')]
    else:
        model_files = ['bla'] #to loop only once TODO find a better workaround

    for model_file in model_files:
        if model_path is not None:
            model.load_weights(model_file)

        form = abs_and_phase if show_abs_and_phase else real_and_imag

        for f in data_files:
            h5f = h5py.File(f, 'r')

            for i in [4]:#range(h5f['tests'].shape[1]):

                #input
                x_r = h5f['inputs'][i,:,:,0]
                x_i = h5f['inputs'][i,:,:,1]
                if not show_kspaces:
                    x_r, x_i = form(ifft2(x_r,x_i))

                #evaluation of model if given one
                if model is not None:
                    output_model = model.predict(np.expand_dims(h5f['inputs'][i],axis=0))

                #intermediate output
                if model is None:
                    y_ks_r = h5f['tests'][0,i,:,:,0]
                    y_ks_i = h5f['tests'][0,i,:,:,1]
                else:
                    y_ks_r = output_model[0][0,:,:,0]
                    y_ks_i = output_model[0][0,:,:,1]


                #intermediate ground truth
                Y_ks_r = h5f['intermediate_outputs'][i,:,:,0]
                Y_ks_i = h5f['intermediate_outputs'][i,:,:,1]
                if show_kspaces:
                    Y_ks_r, Y_ks_i = form(ifft2(Y_ks_r,Y_ks_i))

                #output
                if model is None:
                    if h5f['tests'].shape[-1]==2:
                        y_img_r = h5f['tests'][1,i,:,:,0]
                        y_img_i = h5f['tests'][1,i,:,:,1]
                        if not show_abs_and_phase:
                            y_img_r, y_img_i = form(y_img_r,y_img_i)
                    else:
                        y_img_r = h5f['tests'][1,i,:,:,0]
                        y_img_i = np.zeros(np.shape(y_img_r))
                else:
                    if output_model[1].shape[-1]==2:
                        y_img_r = output_model[1][0,:,:,0]
                        y_img_i = output_model[1][0,:,:,1]
                        if not show_abs_and_phase:
                            y_img_r, y_img_i = form(y_img_r,y_img_i)    
                    else:
                        y_img_r = output_model[1][0,:,:,0]
                        y_img_i = np.zeros(np.shape(y_img_r))

                #ground truth
                if h5f['outputs'].shape[-1]==2:
                    Y_r = h5f['outputs'][i,:,:,0]
                    Y_i = h5f['outputs'][i,:,:,1]
                else:
                    Y_r = h5f['outputs'][i,:,:,0]
                    Y_i = np.zeros(np.shape(Y_r))
                if not show_abs_and_phase:
                    Y_r, Y_i = form(Y_r,Y_i)

                fig, axs = plt.subplots(2,5, figsize=(20, 20))
                fig.subplots_adjust(hspace=0.1, wspace=0.1)
                axs = axs.ravel()
                if show_abs_and_phase:
                    real = 'abs'
                    imag = 'phase'
                else:
                    real = 'real'
                    imag = 'imag'
                axs[0].imshow(x_r,cmap='gray')
                axs[0].title.set_text('KS Input {}'.format(real))
                axs[5].imshow(x_i,cmap='gray')
                axs[5].title.set_text('KS Input {}'.format(imag))
                axs[1].imshow(y_ks_r,cmap='gray')
                axs[1].title.set_text('Intermediate kspace output {}'.format(real))
                axs[6].imshow(y_ks_i,cmap='gray')
                axs[6].title.set_text('Intermediate kspace output {}'.format(imag))
                axs[2].imshow(Y_ks_r,cmap='gray')
                axs[2].title.set_text('Intermediate ground truth {}'.format(real))
                axs[7].imshow(Y_ks_i,cmap='gray')
                axs[7].title.set_text('Intermediate ground truth {}'.format(imag))
                axs[3].imshow(y_img_r,cmap='gray')
                axs[3].title.set_text('output {}'.format(real))
                axs[8].imshow(y_img_i,cmap='gray')
                axs[8].title.set_text('output {}'.format(imag))
                axs[4].imshow(Y_r,cmap='gray')
                axs[4].title.set_text('Ground truth {}'.format(real))
                axs[9].imshow(Y_i,cmap='gray')
                axs[9].title.set_text('Ground truth {}'.format(imag))

                if plot:
                    plt.show()
                else:
                    plt.savefig(model_file[:-2]+'png')
                plt.close()


if __name__ == '__main__':
    tf.config.set_visible_devices([],'GPU')
    model_path = r'D:\NN_DATA\singlecoil_acc30_nonorm_abs&phase_end&interm_onlymiddleslices_test5\trainingsaves_Jan_26_10_15'

    model = get_unet((256,256,2))

    data_file_path = r'D:\NN_DATA\singlecoil_acc30_nonorm_abs&phase_end&interm_onlymiddleslices_test5\train\file1000002.h5'

    show_abs_and_phase = True
    show_kspaces = False

    plot=False
    rewrite=True

    if os.path.isfile(data_file_path) and ('.h5' in data_file_path):
        data_files = [data_file_path]
    else:
        data_files = [os.path.join(data_file_path, f) for f in os.listdir(data_file_path) if (
                os.path.isfile(os.path.join(data_file_path, f)) and ('.h5' in f))]

    plotting(data_files,show_abs_and_phase,show_kspaces,model,model_path,plot=plot,rewrite=rewrite)