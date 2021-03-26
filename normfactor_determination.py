import os
import h5py
import numpy as np
from utils.fastMRI_utils import image_from_kspace, crop, fft, ifft
import matplotlib.pyplot as plt
from plotting import simple_plot
from NN.Generators import DataGenerator_img, DataGenerator_kspace_img, DataGenerator_img_abs
from progress.bar import IncrementalBar


batch_size = 8

train_gen = DataGenerator_img(r'D:\NN_DATA\singlecoil_acc15_absphasekspaceimg_midslices_kabsmaxnorm_pointmask_fullsample\train',batch_size=batch_size)


vmin = -1.
vmax = 1.


# means = []
# means_i = []
# maxs = []
# maxs_i = []
# mins = []
# mins_i = []

# imgmeans = []
# imgmeans_i = []
# imgmaxs = []
# imgmaxs_i = []
# imgmins = []
# imgmins_i = []

bar = IncrementalBar('Dataset_files', max = len(train_gen))

k=0
for batch in train_gen:
    for i in range(batch_size):
        X = batch[0][i]
        # Y_int = batch[1][0][i]
        Y = batch[1][i]

        X_r = X[:,:,0] * np.cos(X[:,:,1])
        X_i = X[:,:,0] * np.sin(X[:,:,1])
        Y_r = Y[:,:,0] * np.cos(Y[:,:,1])
        Y_i = Y[:,:,0] * np.sin(Y[:,:,1])
        # kspace_orig = h5f['kspace'][i,:,:]
        # # simple_plot(kspace_orig)
        # img_orig = ifft(kspace_orig)
        # # simple_plot(img_orig)
        # img_cropped = crop(img_orig, size=(256,256))
        # kspace_in = fft(img_cropped)
        # kspace_in = kspace_in
        # img_in = ifft(kspace_in)

        # kspace_in = X[:,:,0] * (np.cos(X[:,:,1]) + 1j*np.sin(X[:,:,1]))#1j*X[:,:,1]+X[:,:,0]
        # img_in = ifft(kspace_in)
        # img_in = X[:,:,0]

        # means.append(np.mean(np.abs(kspace_in)))
        # means_i.append(np.mean(np.angle(kspace_in)))
        # maxs.append(np.amax(np.abs(kspace_in)))
        # maxs_i.append(np.amax(np.angle(kspace_in)))
        # mins.append(np.amin(np.abs(kspace_in)))
        # mins_i.append(np.amin(np.angle(kspace_in)))

        # imgmeans.append(np.mean(img_in))
        # # imgmeans_i.append(np.mean(np.angle(img_in)))
        # imgmaxs.append(np.amax(img_in))
        # # imgmaxs_i.append(np.amax(np.angle(img_in)))
        # imgmins.append(np.amin(img_in))
        # # imgmins_i.append(np.amin(np.angle(img_in)))
        # # img_in = ifft(kspace_in)

        fig, axs = plt.subplots(2,4)
        # fig.subplots_adjust(hspace=0.1, wspace=0.1)
        axs = axs.ravel()
        axs[0].imshow(X[:,:,0],vmin=vmin,vmax=vmax)
        axs[4].imshow(Y[:,:,0],vmin=vmin,vmax=vmax)
        axs[1].imshow(X[:,:,1],vmin=-1*np.pi,vmax=np.pi)
        axs[5].imshow(Y[:,:,1],vmin=-1*np.pi,vmax=np.pi)
        axs[2].imshow(X_r,vmin=vmin,vmax=vmax)
        axs[6].imshow(Y_r,vmin=vmin,vmax=vmax)
        axs[3].imshow(X_i,vmin=vmin,vmax=vmax)
        axs[7].imshow(Y_i,vmin=vmin,vmax=vmax)
        # axs[0].imshow(np.abs(img_in))
        # axs[1].imshow(np.angle(img_in))
        # axs[2].imshow(Y_int[:,:,0])
        # axs[3].imshow(Y_int[:,:,1])
        # axs[4].imshow(Y[:,:,0])
        # axs[5].imshow(Y[:,:,1])
        plt.savefig(r'Images\im{}.png'.format(k*8+i))
        plt.close()
    bar.next()
    k+=1

# fig, axs = plt.subplots(1,3, figsize=(20, 20))
# # fig.subplots_adjust(hspace=0.1, wspace=0.1)
# axs = axs.ravel()
# axs[0].hist(imgmins)
# axs[0].title.set_text('imgmins abs')
# axs[1].hist(imgmaxs)
# axs[1].title.set_text('imgmaxs abs')
# axs[2].hist(imgmeans)
# axs[2].title.set_text('imgmeans abs')

# # axs[0].hist(means)
# # axs[0].title.set_text('means abs')
# # axs[6].hist(means_i)
# # axs[6].title.set_text('means angle')
# # axs[1].hist(maxs)
# # axs[1].title.set_text('maxs abs')
# # axs[7].hist(maxs_i)
# # axs[7].title.set_text('maxs angle')
# # axs[2].hist(mins)
# # axs[2].title.set_text('mins abs')
# # axs[8].hist(mins_i)
# # axs[8].title.set_text('mins angle')
# # axs[3].hist(imgmeans)
# # axs[3].title.set_text('imgmeans abs')
# # axs[9].hist(imgmeans_i)
# # axs[9].title.set_text('imgmeans angle')
# # axs[4].hist(imgmaxs)
# # axs[4].title.set_text('imgmaxs abs')
# # axs[10].hist(imgmaxs_i)
# # axs[10].title.set_text('imgmaxs angle')
# # axs[5].hist(imgmins)
# # axs[5].title.set_text('imgmins abs')
# # axs[11].hist(imgmins_i)
# # axs[11].title.set_text('imgmins angle')
# for ax in axs:
#     ax.set_yscale('log')
# plt.savefig('singlecoil_stats.png')
