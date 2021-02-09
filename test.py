from NN.architectures import get_unet, nrmse_2D_L2, get_wnet, get_unet_fft
from tensorflow.keras.optimizers import Adam
from NN.Inputs import DataGenerator_img, DataGenerator_kspace_img, DataGenerator_kspace_to_img
import tensorflow as tf
import matplotlib.pyplot as plt
from plotting import ifft2
import numpy as np

def ifft_from_abs_angle(absim,angleim):
    realim = absim * np.cos(angleim)
    imim = absim * np.sin(angleim)
    im = ifft2(realim,imim)
    return np.abs(im), np.angle(im) 

tf.config.set_visible_devices([],'GPU')

model = get_unet_fft((256,256,2),kernel_initializer="zeros", fullskip=True)

model.compile(loss = nrmse_2D_L2,optimizer=Adam())

train_gen = DataGenerator_kspace_to_img(r'D:\NN_DATA\singlecoil_acc15_absphasekspaceimg_midslices_kabsmaxnorm\train', batch_size=1)


# absim = output[0,:,:,0]
# angleim = output[0,:,:,1]
# absim, angleim = ifft_from_abs_angle(absim,angleim)


abstru = train_gen[0][0][0,:,:,0]
angtru = train_gen[0][0][0,:,:,1]
abstru, angtru = ifft_from_abs_angle(abstru,angtru)
output = model.predict(train_gen[0])

absim = output[0,:,:,0]
angleim = output[0,:,:,1]
fig, axs = plt.subplots(3,2, figsize=(20, 20))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
axs = axs.ravel()
axs[0].imshow(absim)
axs[1].imshow(angleim)
axs[2].imshow(train_gen[0][1][0,:,:,0])
axs[3].imshow(train_gen[0][1][0,:,:,1])
axs[4].imshow(abstru)
axs[5].imshow(angtru)
plt.show()