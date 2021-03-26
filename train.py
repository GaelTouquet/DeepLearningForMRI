from NN.Agent import Agent
from NN.Inputs import prepare_datasets
from NN.Generators import find_generator, DataGenerator_kspace_img_interm_kspace_onlyabsimg
from NN.Masks import RandomMask, CenteredRandomMask, PolynomialMaskGenerator
from NN.architectures import nrmse, nrmse_2D_L2, nrmse_2D_L1, norm_abs, unnorm_abs, my_ssim, reduced_nrmse
from NN.architectures import reconGAN_Unet, reconGAN_Unet_kspace_to_img, reconGAN_Wnet, reconGAN_Wnet_intermediate
from plotting import model_output_plotting
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import model_from_json
import numpy as np
import os
import numpy as np
import time
from progress.bar import IncrementalBar

np.random.seed(123)  # for reproducibility

def write_log(callback, names, logs, batch_no,logDir):
    writer = tf.summary.create_file_writer(logDir)
    for name, value in zip(names, logs):
        with writer.as_default():
            tf.summary.scalar(name, value, step=batch_no)
            writer.flush()
### user settings ###

# paths
fast_mri_path = 'D:\\fastMRI_DATA'
base_work_path = 'D:\\NN_DATA'

# data
coil_type = 'single' # 'single' or 'multi'
fraction = 1 # fraction of training sample used, 1 = all, 0.5 = half the training sample
validation_fraction = 1
n_slice_per_file=10
acceleration = 8
sampling_factor = 1./acceleration # fraction of kspace that is measured (=not hidden)

normalise = False# wether to normalize or not, TODO change this to normalisation factor
image_shape = (256,256)
n_channels_in = 1
n_channels_intermediate = 1
n_channels_out = 1

# network

batch_size = 16
epochs = 1000
n_tests = 1
# tags for the run
normalise_image= True
center_normalised_values= True
input_kspace = True
output_image = True # don't forget to change arch to one that contains ifft if needed, not automatic yet
intermediate_output = 'kspace'
absphase_img = False
absphase_kspace = False

datatag = '_'.join([
    'ksap' if absphase_kspace else 'ksri',
    'imgap' if absphase_img else 'imgri',
    'normed' if normalise_image else '',
    '{}midslices'.format(n_slice_per_file),
    'densedpointmasked_nofftshift2'
])

# arch_type = 'ReconGAN_Unet'
loss=my_ssim
loss_weights = [0.1,0.9]#[0.01,0.99]
optimizer=Adam(learning_rate=1e-4)
generator_type = find_generator(input_kspace,output_image,intermediate_output)
dropout = 0.05

indim = image_shape
outdim = image_shape

# model = reconGAN_Wnet((*image_shape,2), 16, 16, skip=True,realimag_img=not absphase_img, realimag_kspace= not absphase_kspace,normalise_image=normalise_image,center_normalised_values=center_normalised_values)
model = reconGAN_Wnet_intermediate((*image_shape,2), 16, 16, skip=True,realimag_img=not absphase_img, realimag_kspace= not absphase_kspace,normalise_image=normalise_image,center_normalised_values=center_normalised_values,dropout=dropout)
# model = reconGAN_Unet_kspace_to_img((*image_shape,2), 16,skip=True,realimag_img=not absphase_img, realimag_kspace= not absphase_kspace,normalise_image=normalise_image,center_normalised_values=center_normalised_values)
# model = reconGAN_Unet((*image_shape,2), 16,skip=True)

agenttag = '_'.join([
    'ReconGAN_Unet',# if model in [reconGAN_Unet,reconGAN_Unet_kspace_to_img] else 'ReconGAN_Wnet',
    'kspace' if input_kspace else 'img',
    'to',
    'img' if output_image else 'kspace',
    'intermoutput' if intermediate_output else '',
    'ssim'#'ssim'
])

input_mask = PolynomialMaskGenerator(image_shape,sampling_factor=sampling_factor,dim=2)#CenteredRandomMask(acceleration=acceleration, center_fraction=(4./100.), seed=0xdeadbeef)#

# model = reconGAN((256,256,2),16,skip=False)#get_unet(input_shape=(*indim,n_channels_in),depth=8,n_filters=4,batchnorm=False)#get_unet_fft(input_shape=(*indim,n_channels_in),fullskip=True,normfunction=norm_abs, unormfunc=unnorm_abs,depth=6,kernel=2,n_filters=16,batchnorm=False,dropout=0)#get_wnet(input_shape=(*indim,n_channels_in),kernel_initializer=RandomNormal())#not_complex#complex_but_not
# model = testmodel()
# json_file = open(r'D:\NN_DATA\singlecoil_acc15_absphasekspaceimg_midslices_kabsmaxnorm\trainingsaves_Unet_kspacetoimg_absnormunnorm_depth9kernel2_nobatchnorm_Feb_09_10_55\model_save.json', 'r')
# model = model_from_json(json_file.read())
# json_file.close()
# model.load_weights(r'D:\NN_DATA\singlecoil_acc15_absphasekspaceimg_midslices_kabsmaxnorm\trainingsaves_Unet_kspacetoimg_absnormunnorm_depth9kernel2_nobatchnorm_Feb_09_10_55\best.h5')



model.compile(loss = [reduced_nrmse,tf.keras.losses.MeanAbsoluteError()],optimizer=optimizer, loss_weights=loss_weights)#,tf.keras.losses.MeanAbsoluteError() loic
# model.load_weights(r'D:\NN_DATA\singlecoil_acc15_absphasekspaceimg_10midslices_kabsmaxnormnomin_pointmask\trainingsaves_ReconGANWnet_abs&phase_skip_ssim_Mar_02_19_19\best.h5')



# model.load_weights(r'D:\NN_DATA\singlecoil_acc15_imgnorm_absphasekspaceimg_midslices_kabsmaxnorm\trainingsaves_ReconGAN_absimg_Feb_12_15_06\best.h5')


timestamp = time.strftime("%h_%d_%H_%M")

name = '{}coil_acc{}_{}'.format(coil_type,int(sampling_factor*100),datatag)
workdir = os.path.join(base_work_path,name)
print('\nusing Directory :\n{}\n'.format(workdir))
if not os.path.isdir(workdir):
    os.mkdir(workdir)

# model.load_weights(r'D:\NN_DATA\singlecoil_acc30_nonorm_abs&phase_end&interm_onlymiddleslices_test5\trainingsaves_Jan_25_17_12\epoch200.h5')
# model = tf.keras.models.load_model(r'D:\NN_DATA\singlecoil_acc30_onlygoodslices_nonorm\agentJan_19_15_40')


train_path = prepare_datasets(datapath=os.path.join(fast_mri_path,'{}coil_train'.format(coil_type)),
                workdirpath=workdir, dataset_type='train', 
                input_mask=input_mask, fraction=fraction,
                image_shape=image_shape,
                n_slice_per_file=n_slice_per_file,
                absphase_img=absphase_img,absphase_kspace=absphase_kspace,
                normalise_image=normalise_image,
                center_normalised_values=center_normalised_values)
val_path = prepare_datasets(datapath=os.path.join(fast_mri_path,'{}coil_val'.format(coil_type)),
                workdirpath=workdir, dataset_type='val', #timestamp+
                input_mask=input_mask, fraction=fraction,
                image_shape=image_shape,
                n_slice_per_file=n_slice_per_file,
                absphase_img=absphase_img,absphase_kspace=absphase_kspace,
                normalise_image=normalise_image,
                center_normalised_values=center_normalised_values)


print('preparing generators')
train_gen = generator_type(train_path, batch_size=batch_size)
val_gen = generator_type(val_path, batch_size=batch_size)

# from generatorKspaceUnet3 import DataGenerator
# train_gen = DataGenerator([r'D:\LB-NN-MONICA-KSPACE\Train-k-space-monica-15.hdf5'],batch_size=batch_size, dim=(256,256)) #loic
# val_gen = DataGenerator([r'D:\LB-NN-MONICA-KSPACE\Val-k-space-monica-15.hdf5'],batch_size=batch_size, dim=(256,256))


train_dir = os.path.join(workdir,'trainingsaves_{}_{}'.format(agenttag,timestamp))
os.mkdir(train_dir)

print('Saving model')
model_json = model.to_json()
with open(os.path.join(train_dir,'model_save.json'),'w') as json_file:
    json_file.write(model_json)


tensorboard_callback = TensorBoard(log_dir=os.path.join(train_dir,'log'))
save_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(train_dir,'epoch{epoch:02d}.h5'), save_freq='epoch', save_best_only=False,save_weights_only=True)
save_best_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(train_dir,'best.h5'), save_freq='epoch', save_best_only=True,save_weights_only=True)
# first=True
# last20_weights = []
# last20_batches = []
# last20_loss = []
# last_log = 2.
# print('Training')
# for i, epoch in enumerate(range(epochs)):
#     bar = IncrementalBar('Batch',max=len(train_gen))
#     for k,batch in enumerate(train_gen):
#         logs = model.train_on_batch(batch[0],batch[1])
#         # print(logs)
#         # if any([np.any(np.isnan(layer.numpy())) for layer in model.weights]) or logs>1:
#         #     print('nan from here')
#         #     import pdb;pdb.set_trace()
#         #     first=False
#         write_log(tensorboard_callback,['train_loss'],[logs],i,os.path.join(train_dir,'log'))
#         model.save_weights(os.path.join(train_dir,'epoch{epoch:02d}_batch{batch:02d}.h5'.format(epoch=i,batch=k)))
#         if logs > 1.2*last_log:
#             print('found huge loss gain, skipping batch')
#             if k<5:
#                 model.load_weights(os.path.join(train_dir,'epoch{epoch:02d}_batch{batch:02d}.h5'.format(epoch=i-1,batch=len(train_gen)-1)))
#             else:
#                 model.load_weights(os.path.join(train_dir,'epoch{epoch:02d}_batch{batch:02d}.h5'.format(epoch=i,batch=k-5)))
#             continue
#         last_log = logs
#         # last20_weights.append([layer.numpy() for layer in model.weights])
#         # last20_batches.append([batch[0],batch[1]])
#         # last20_loss.append(logs)
#         bar.next()
#     # model.save_weights(os.path.join(train_dir,'epoch{epoch:02d}.h5'.format(epoch=i))

training_parameters = {

}

history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[tensorboard_callback,save_callback,save_best_callback])#

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


###plotting
model_path = train_dir
data_file_path = '\\'.join(model_path.split('\\')[:-1]+['val','file1000000.h5'])
data_files = [data_file_path]

# needed from data/model
input_image = not input_kspace
intermediate_output = intermediate_output
ifft_output = not output_image
ir_kspace = not absphase_kspace
ir_img = not absphase_img

# plotting options
display_real_imag = not absphase_img

model_output_plotting(data_files,model,model_path,plot=False,rewrite=True, only_best=False, intermediate_output=intermediate_output, display_real_imag=display_real_imag, input_image=input_image, ifft_output=ifft_output, ir_kspace=ir_kspace,ir_img=ir_img)