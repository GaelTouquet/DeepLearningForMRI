from NN.Agent import Agent
from NN.Inputs import prepare_datasets
from NN.Generators import DataGenerator_kspace_img, DataGenerator_img, DataGenerator_kspace, DataGenerator_kspace_to_img
from NN.Masks import RandomMask, CenteredRandomMask, PolynomialMaskGenerator
from NN.architectures import get_wnet, get_unet, nrmse, nrmse_2D_L2, nrmse_2D_L1, testmodel, get_unet_fft, norm_abs, unnorm_abs, simple_dense
# from plotting import plotting
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import model_from_json
import numpy as np
import os
import numpy as np
import time
np.random.seed(123)  # for reproducibility

### user settings ###

# paths
fast_mri_path = 'D:\\fastMRI_DATA'
base_work_path = 'D:\\NN_DATA'

# data
coil_type = 'single' # 'single' or 'multi'
fraction = 1 # fraction of training sample used, 1 = all, 0.5 = half the training sample
validation_fraction = 1
sampling_factor = 0.15 # fraction of kspace that is measured (=not hidden)
normalise = False# wether to normalize or not, TODO change this to normalisation factor
image_shape = (256,256)
n_channels_in = 2
n_channels_intermediate = 2
n_channels_out = 2

indim = image_shape
outdim = image_shape
input_mask = PolynomialMaskGenerator(image_shape,sampling_factor=sampling_factor)#CenteredRandomMask(acceleration=acceleration, center_fraction=(24/360), seed=0xdeadbeef)

# network

batch_size = 8
epochs = 1
model = get_unet_fft(input_shape=(*indim,n_channels_in),fullskip=True,normfunction=norm_abs, unormfunc=unnorm_abs,depth=6,kernel=2,n_filters=16,batchnorm=False,dropout=0)#get_wnet(input_shape=(*indim,n_channels_in),kernel_initializer=RandomNormal())#not_complex#complex_but_not
# model = testmodel()
# json_file = open(r'D:\NN_DATA\singlecoil_acc15_absphasekspaceimg_midslices_kabsmaxnorm\trainingsaves_Unet_kspacetoimg_absnormunnorm_depth9kernel2_nobatchnorm_Feb_09_10_55\model_save.json', 'r')
# model = model_from_json(json_file.read())
# json_file.close()
# model.load_weights(r'D:\NN_DATA\singlecoil_acc15_absphasekspaceimg_midslices_kabsmaxnorm\trainingsaves_Unet_kspacetoimg_absnormunnorm_depth9kernel2_nobatchnorm_Feb_09_10_55\best.h5')

model.compile(loss = nrmse_2D_L1,optimizer=Adam())#,tf.keras.losses.MeanAbsoluteError() 
# model.load_weights(r'D:\NN_DATA\singlecoil_acc15_absphasekspaceimg_midslices_kabsmaxnorm\trainingsaves_Unet_img_absnormunnorm_depth8_nobatchnorm_Feb_05_16_26\epoch50.h5')

# tags for the run
datatag = 'absphasekspaceimg_midslices_kabsmaxnorm' # if already run with this tag, same data is used (=data will not be re-generated)
agenttag = 'Unet_kspacetoimg_absnormunnorm_depth9kernel2_nobatchnorm'


timestamp = time.strftime("%h_%d_%H_%M")

name = '{}coil_acc{}_{}'.format(coil_type,int(sampling_factor*100),datatag)
workdir = os.path.join(base_work_path,name)
print('\nusing Directory :\n{}\n'.format(workdir))
if not os.path.isdir(workdir):
    os.mkdir(workdir)

# model.load_weights(r'D:\NN_DATA\singlecoil_acc30_nonorm_abs&phase_end&interm_onlymiddleslices_test5\trainingsaves_Jan_25_17_12\epoch200.h5')
# model = tf.keras.models.load_model(r'D:\NN_DATA\singlecoil_acc30_onlygoodslices_nonorm\agentJan_19_15_40')


print('preparing train data')
train_path = prepare_datasets(datapath=os.path.join(fast_mri_path,'{}coil_train'.format(coil_type)),
                workdirpath=workdir, dataset_type='train', 
                input_mask=input_mask, fraction=fraction,
                image_shape=image_shape,
                n_slice_per_file=10)
print('preparing val data')
val_path = prepare_datasets(datapath=os.path.join(fast_mri_path,'{}coil_val'.format(coil_type)),
                workdirpath=workdir, dataset_type='val', #timestamp+
                input_mask=input_mask, fraction=fraction,
                image_shape=image_shape,
                n_slice_per_file=10)


print('preparing generators')
train_gen = DataGenerator_kspace_to_img(train_path, batch_size=batch_size)
val_gen = DataGenerator_kspace_to_img(val_path, batch_size=batch_size)

#callbacks
train_dir = os.path.join(workdir,'trainingsaves_{}_{}'.format(agenttag,timestamp))
os.mkdir(train_dir)
tensorboard_callback = TensorBoard(log_dir=os.path.join(train_dir,'log'), histogram_freq=1)
save_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(train_dir,'epoch{epoch:02d}.h5'), save_freq='epoch', save_best_only=False,save_weights_only=True)
save_best_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(train_dir,'best.h5'), save_freq='epoch', save_best_only=True,save_weights_only=True)

print('Training')
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[tensorboard_callback,save_callback,save_best_callback])#

print('Saving model')
model_json = model.to_json()
with open(os.path.join(train_dir,'model_save.json'),'w') as json_file:
    json_file.write(model_json)
# model.save(os.path.join(train_dir,'model_save'))


# myagent = Agent(model, train_gen, val_gen, workdir, timestamp)
# print('starting training')
# myagent.train(epochs=epochs, verbose=1)
# print('saving agent')
# myagent.save(os.path.join(workdir,'agent{}'.format(timestamp)))
# print('making plots')
# data_file_path = os.path.join(workdir,'val_{}'.format(agenttag))#timestamp+
# data_files = [os.path.join(data_file_path, f) for f in os.listdir(data_file_path) if (
#             os.path.isfile(os.path.join(data_file_path, f)) and ('.h5' in f))]
# plotting(data_files,show_abs_and_phase=True,show_kspaces=False,model=myagent.network,model_path=os.path.join(workdir,'trainingsaves_{}'.format(timestamp)),plot=False)