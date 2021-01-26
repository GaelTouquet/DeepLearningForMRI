from NN.Agent import Agent
from NN.Inputs import prepare_datasets, DataGenerator,  test_data
from NN.Masks import RandomMask, CenteredRandomMask, PolynomialMaskGenerator
from NN.architectures import get_unet, test_model, nrmse, nrmse_2D#complex_but_not, not_complex
from plotting import plotting
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
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
sampling_factor = 0.30 # fraction of kspace that is measured (=not hidden)
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
epochs = 200
model = get_unet(input_shape=(*indim,n_channels_in),kernel_initializer=RandomNormal())#not_complex#complex_but_not

# tags for the run
datatag = 'nonorm_abs&phase_end&interm_onlymiddleslices_test5' # if already run with this tag, same data is used (=data will not be re-generated)
agenttag = 'abs&phase_Wnet_randnorminitstd1e-6_baselearningrate'



### Main code ###

timestamp = time.strftime("%h_%d_%H_%M")

name = '{}coil_acc{}_{}'.format(coil_type,int(sampling_factor*100),datatag)
workdir = os.path.join(base_work_path,name)
print('\nusing Directory :\n{}\n'.format(workdir))
if not os.path.isdir(workdir):
    os.mkdir(workdir)

print("compiling model")
model.compile(loss = [nrmse_2D,nrmse_2D],optimizer=Adam())#,tf.keras.losses.MeanAbsoluteError() 

model.load_weights(r'D:\NN_DATA\singlecoil_acc30_nonorm_abs&phase_end&interm_onlymiddleslices_test5\trainingsaves_Jan_25_17_12\epoch200.h5')
# model = tf.keras.models.load_model(r'D:\NN_DATA\singlecoil_acc30_onlygoodslices_nonorm\agentJan_19_15_40')



print('preparing train data')
train_indexpath = prepare_datasets(datapath=os.path.join(fast_mri_path,'{}coil_train'.format(coil_type)),
                workdirpath=workdir, dataset_type='train', 
                input_mask=input_mask,
                multicoil=(coil_type=='multi'),
                normalise=normalise,fraction=fraction,
                input_shape=image_shape,
                n_channels_in=n_channels_in,
                n_channels_intermediate=n_channels_intermediate,
                n_channels_out=n_channels_out,
                n_slice_per_file=10)
print('preparing val data')
val_indexpath = prepare_datasets(datapath=os.path.join(fast_mri_path,'{}coil_val'.format(coil_type)),
                workdirpath=workdir, dataset_type='val_{}'.format(agenttag), #timestamp+
                input_mask=input_mask,
                multicoil=(coil_type=='multi'),
                normalise=normalise,fraction=fraction,
                input_shape=image_shape,
                n_channels_in=n_channels_in,
                n_channels_intermediate=n_channels_intermediate,
                n_channels_out=n_channels_out,
                n_slice_per_file=10)
# prepare_datasets(os.path.join(fast_mri_path,'{}coil_val'.format(coil_type)),
#                 name, 'val_{}'.format(timestamp+agenttag), input_mask=input_mask,multicoil=(coil_type=='multi'),normalise=normalise,fraction=validation_fraction,input_shape=image_shape,n_channels_in=n_channels_in,n_channels_out=n_channels_out)


print('preparing generators')
train_gen = DataGenerator(train_indexpath,
    batch_size=batch_size,indim=indim,outdim=outdim,n_channels_in=n_channels_in,n_channels_out=n_channels_out)
val_gen = DataGenerator(val_indexpath, 
    batch_size=batch_size,indim=indim,outdim=outdim,n_channels_in=n_channels_in,n_channels_out=n_channels_out)
# test_gen = DataGenerator(name.format('val'),batch_size=batch_size)


myagent = Agent(model, train_gen, val_gen, workdir, timestamp)
print('starting training')
myagent.train(epochs=epochs, verbose=1)
# print('starting testing')
# test_data(os.path.join(workdir,'val_{}'.format(timestamp+agenttag)),myagent,batch_size=batch_size)
print('saving agent')
myagent.save(os.path.join(workdir,'agent{}'.format(timestamp)))
print('making plots')
data_file_path = os.path.join(workdir,'val_{}'.format(agenttag))#timestamp+
data_files = [os.path.join(data_file_path, f) for f in os.listdir(data_file_path) if (
            os.path.isfile(os.path.join(data_file_path, f)) and ('.h5' in f))]
plotting(data_files,show_abs_and_phase=True,show_kspaces=False,model=myagent.network,model_path=os.path.join(workdir,'trainingsaves_{}'.format(timestamp)),plot=False)