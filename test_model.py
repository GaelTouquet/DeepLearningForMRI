import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import mean_squared_error
from NN.Generators import DataGenerator_img_abs
import tensorflow as tf
import numpy as np
from progress.bar import IncrementalBar
import pickle
import matplotlib.pyplot as plt


tf.config.set_visible_devices([],'GPU')

np.random.seed(0)

model_path = r'D:\NN_DATA\singlecoil_acc15_imgnorm_absphasekspaceimg_midslices_kabsmaxnorm\trainingsaves_ReconGANUnet_abs&phase_nanvaltest_Feb_15_08_42'

data_path = r'D:\NN_DATA\singlecoil_acc15_imgnorm_absphasekspaceimg_midslices_kabsmaxnorm\train'

json_file = os.path.join(model_path,'model_save.json')
if os.path.isfile(json_file):
    json_file = open(json_file, 'r')
    model = model_from_json(json_file.read())
    json_file.close()

model_weights_path = os.path.join(model_path,'epoch00_batch01.h5')

model.load_weights(model_weights_path)

generator = DataGenerator_img_abs(data_path, batch_size=1)

# bar = IncrementalBar('Image', max = len(generator))

mse_values = []

# if not os.path.isfile('values'):
weight_stats={'min':[],'max':[],'absmin':[],'mean':[]}
first=True
for X,Y in generator:
    y = model.predict(X)
    loss = np.mean(mean_squared_error(Y[0,:,:,0],y[0,:,:,0]))
    m = 6150
    k=0
    while m<6155:#True:#np.isnan(loss) or np.isinf(loss) or loss<=0. or loss >0.07:
        m+=1
        # if m==-1:
        #     k-=1
        #     m=1215
        model_weights_path = os.path.join(model_path,'epoch{epoch:02d}_batch{batch:02d}.h5'.format(epoch=k,batch=m))
        model.load_weights(model_weights_path)
        y = model.predict(X)
        loss = np.mean(mean_squared_error(Y[0,:,:,0],y[0,:,:,0]))
        print('\nimage {}'.format(m))
        print(loss)
        print(min([tf.math.reduce_min(layer).numpy() for layer in model.weights]))
        print(min([tf.math.reduce_min(tf.math.abs(layer)).numpy() for layer in model.weights]))
        print(max([tf.math.reduce_max(layer).numpy() for layer in model.weights]))
        if np.isnan(loss) and first:
            print('first nan: ',m)
            first=False
            import pdb;pdb.set_trace()
        print()
        # weight_stats['min'].append(min([tf.math.reduce_min(layer).numpy() for layer in model.weights]))
        # weight_stats['absmin'].append(min([tf.math.reduce_min(tf.math.abs(layer)).numpy() for layer in model.weights]))
        # weight_stats['max'].append(max([tf.math.reduce_max(layer).numpy() for layer in model.weights]))
        # weight_stats['mean'].append([tf.math.reduce_mean(layer).numpy for layer in model.weights])
    import pdb;pdb.set_trace()
    mse_values.append(loss)
    # bar.next()

# with open('values') as fb:
#     mse_values = pickle.load(fb)


plt.hist(mse_values)
plt.show()




# from tensorflow.keras.layers import Conv2D, concatenate,Input, Add, Conv2DTranspose
# from tensorflow.keras.models import Model
# ## test from article


# reconGAN((256,256,2),16)
# from NN.architectures import get_unet, nr

# from tensorflow.keras.optimizers import Adam
# from NN.Inputs import DataGenerator_img, DataGenerator_kspace_img, DataGenerator_kspace_to_img
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from plotting import ifft2
# import numpy as np

# def ifft_from_abs_angle(absim,angleim):
#     realim = absim * np.cos(angleim)
#     imim = absim * np.sin(angleim)
#     im = ifft2(realim,imim)
#     return np.abs(im), np.angle(im) 

# tf.config.set_visible_devices([],'GPU')

# model = get_unet_fft((256,256,2),kernel_initializer="zeros", fullskip=True)

# model.compile(loss = nrmse_2D_L2,optimizer=Adam())

# train_gen = DataGenerator_kspace_to_img(r'D:\NN_DATA\singlecoil_acc15_absphasekspaceimg_midslices_kabsmaxnorm\train', batch_size=1)


# # absim = output[0,:,:,0]
# # angleim = output[0,:,:,1]
# # absim, angleim = ifft_from_abs_angle(absim,angleim)


# abstru = train_gen[0][0][0,:,:,0]
# angtru = train_gen[0][0][0,:,:,1]
# abstru, angtru = ifft_from_abs_angle(abstru,angtru)
# output = model.predict(train_gen[0])

# absim = output[0,:,:,0]
# angleim = output[0,:,:,1]
# fig, axs = plt.subplots(3,2, figsize=(20, 20))
# fig.subplots_adjust(hspace=0.1, wspace=0.1)
# axs = axs.ravel()
# axs[0].imshow(absim)
# axs[1].imshow(angleim)
# axs[2].imshow(train_gen[0][1][0,:,:,0])
# axs[3].imshow(train_gen[0][1][0,:,:,1])
# axs[4].imshow(abstru)
# axs[5].imshow(angtru)
# plt.show()