from NN.Agent import Agent
from NN.Inputs import prepare_datasets
from NN.Generators import DataGenerator
# from NN.Generators import find_generator, DataGenerator_kspace_img_interm_kspace_onlyabsimg, DataGenerator_complex
from NN.Masks import RandomMask, CenteredRandomMask, PolynomialMaskGenerator, MaskHandler, CentralMask
from plotting import model_output_plotting
import tensorflow as tf
from utils.normalisation import normalisation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import model_from_json
from tensorflow.keras.activations import relu
from keras.utils.vis_utils import plot_model
import pickle
import numpy as np
import os
import numpy as np
import time
from progress.bar import IncrementalBar
from utils.utils import get_git_revisions_hash

np.random.seed(123)  # for reproducibility

### USER PARAMETERS ###

params = {
    # name
    'name' : 'Wnet_perf',
    # git commit
    'commit' : get_git_revisions_hash(),
    # paths
    'fast_mri_path' : 'D:\\fastMRI_DATA',
    'base_work_path' : 'D:\\NN_DATA',
    # data
    'n_coil' : 15,
    'dataset_fraction' : 1, # fraction of training sample used, 1 = all, 0.5 = half the training sample
    'n_slice_per_file' : 10,
    'sampling_factor' : 0.15,
    'normalisation' : False,# wether to normalize or not, TODO change this to normalisation factor
    'image_shape' : (256,256),
    'img_norm_name' : 'absnorm',
    'img_norm_spe' : 'nponly',
    'kspace_norm_name' : '',
    'kspace_norm_spe' : '',
    # network
    'batch_size' : 16,
    'epochs' : 200,
    'kernel_size' : (5,5),
    'input_kspace' : True,
    'output_image' : True,
    'intermediate_output' : 'kspace',#'kspace',#'kspace',#'kspace','image'
    'mask_kspace' : True,
    'realimag_img' : True,
    'realimag_kspace' : True,
    'is_complex' : True,
    'agent_tag' : 'multicoil_test',
    'data_tag' : 'multicoil_test',
}

### SOME MORE PARAMETERS AFTER ###
if params['n_coil'] is not None:
    from NN.architectures_coil import my_ssim, reduced_nrmse
    from NN.architectures_coil import reconGAN_Unet, reconGAN_Unet_kspace_to_img, reconGAN_Wnet, reconGAN_Wnet_intermediate, dense_kspace, dense_kspace_img_out
else:
    if params['is_complex']:
        from NN.architectures_cplx import my_ssim, reduced_nrmse
        from NN.architectures_cplx import reconGAN_Unet, reconGAN_Unet_kspace_to_img, reconGAN_Wnet, reconGAN_Wnet_intermediate, dense_kspace, dense_kspace_img_out
    else:
        from NN.architectures import nrmse, nrmse_2D_L2, nrmse_2D_L1, norm_abs, unnorm_abs, my_ssim, reduced_nrmse
        from NN.architectures import reconGAN_Unet, reconGAN_Unet_kspace_to_img, reconGAN_Wnet, reconGAN_Wnet_intermediate

img_norm = normalisation(name=params['img_norm_name'],spe=params['img_norm_spe'])
kspace_norm = normalisation(name=params['kspace_norm_name'],spe=params['kspace_norm_spe'])

# input_shape = (*params['image_shape'],2,1) if params['is_complex'] else (*params['image_shape'],2)
input_shape = (*params['image_shape'],params['n_coil'],2,1) 
params['input_shape'] = input_shape

params['coil_type'] = 'single' if params['n_coil'] == 1 else 'multi'
### MORE USER PARAMETERS ###

# input_mask = PolynomialMaskGenerator(image_shape,sampling_factor=sampling_factor,dim=2)#CenteredRandomMask(acceleration=acceleration, center_fraction=(4./100.), seed=0xdeadbeef)#
# input_mask = MaskHandler(r'D:\MRI_Masks\0p15\masks.h5')
masked_shape = None#(40,40)
# masked_shape=None
input_mask = MaskHandler(r'D:\MRI_Masks\0p15\masks.h5')
# input_mask = PolynomialMaskGenerator(params['image_shape'],sampling_factor=params['sampling_factor'],dim=1,poly=8,keep_mask=True)#CenteredRandomMask(acceleration=acceleration, center_fraction=(4./100.), seed=0xdeadbeef)#
# input_mask = CentralMask(masked_shape)


loss=['mae',my_ssim]#'mae', 'mse', reduced_nrmse, my_ssim, tf.keras.losses.MeanAbsoluteError()
loss_weights = [0.001,0.999]#None
optimizer=Adam(learning_rate=1e-4)
dropout = 0.05

# model = reconGAN_Wnet(input_shape, 16, 16, skip=True,realimag_img=params['realimag_img'], realimag_kspace=params['realimag_kspace'],normalise_image=normalise_image,center_normalised_values=center_normalised_values)
model = reconGAN_Wnet_intermediate(input_shape, 16, 16, skip_image=True, kernel_size=params['kernel_size'],realimag_img=params['realimag_img'], realimag_kspace=params['realimag_kspace'],img_norm=img_norm,dropout=dropout,kernel_initializer='glorot_normal',mask_kspace=params['mask_kspace'])
# model = reconGAN_Unet_kspace_to_img(input_shape, 16,skip=True, kernel_size=params['kernel_size'],realimag_img=params['realimag_img'], realimag_kspace=params['realimag_kspace'],img_norm=img_norm,mask_kspace=params['mask_kspace'],kernel_initializer='glorot_normal')
# model = reconGAN_Unet(input_shape, 8,skip=False,depth=5,dropout=dropout,realimag_img=params['realimag_img'], realimag_kspace=params['realimag_kspace'],img_norm=img_norm, activation=relu, output_shape=input_shape,mask_kspace=params['mask_kspace'])
# model = dense_kspace(full_masked_shape,input_shape,mask_kspace=params['mask_kspace'])
# model = dense_kspace_img_out(full_masked_shape,input_shape,realimag_img=params['realimag_img'], realimag_kspace=params['realimag_kspace'],img_norm=img_norm,mask_kspace=True)

### END OF USER PARAMETERS ###


params['model_name'] = model.name
params['mask_name'] = input_mask.__class__.__name__ if input_mask else 'NoMask'


datatag = '_'.join([
    'ksri' if params['realimag_img'] else 'ksap',
    'imgri' if params['realimag_kspace'] else 'imgap',
    '{}midslices'.format(params['n_slice_per_file']),
    params['data_tag']
])

agenttag = '_'.join([
    'kspace' if params['input_kspace'] else 'img',
    'to',
    'img' if params['output_image'] else 'kspace',
    'intermoutput' if params['intermediate_output'] else '',
    'complex' if params['is_complex'] else 'real',
    params['agent_tag']
])



model.compile(loss = loss,optimizer=optimizer, loss_weights=loss_weights)

timestamp = time.strftime("%h_%d_%H_%M")

name = '{}coil_acc{}_{}'.format(params['coil_type'],int(params['sampling_factor']*100),datatag)
params['datadir'] = os.path.join(params['base_work_path'],name)
print('\nusing data directory :\n{}\n'.format(params['datadir']))
if not os.path.isdir(params['datadir']):
    os.mkdir(params['datadir'])


train_path = prepare_datasets(datapath=os.path.join(params['fast_mri_path'],'{}coil_train'.format(params['coil_type'])),
                workdirpath=params['datadir'], dataset_type='train', 
                input_mask=input_mask, fraction=params['dataset_fraction'],
                image_shape=params['image_shape'],
                n_slice_per_file=params['n_slice_per_file'],
                realimag_img=params['realimag_img'],realimag_kspace=params['realimag_kspace'],
                kspace_norm=kspace_norm,
                img_norm=img_norm, post_mask_shape=masked_shape,ncoil=params['n_coil'])
val_path = prepare_datasets(datapath=os.path.join(params['fast_mri_path'],'{}coil_val'.format(params['coil_type'])),
                workdirpath=params['datadir'], dataset_type='val', #timestamp+
                input_mask=input_mask, fraction=params['dataset_fraction'],
                image_shape=params['image_shape'],
                n_slice_per_file=params['n_slice_per_file'],
                realimag_img=params['realimag_img'],realimag_kspace=params['realimag_kspace'],
                kspace_norm=kspace_norm,
                img_norm=img_norm, post_mask_shape=masked_shape,ncoil=params['n_coil'])

train_gen = DataGenerator(train_path, input_shape, input_kspace=params['input_kspace'],output_image=params['output_image'],
    intermediate_output=params['intermediate_output'],batch_size=params['batch_size'],
    mask=params['mask_kspace'],ncoil=params['n_coil'])
val_gen = DataGenerator(val_path, input_shape, input_kspace=params['input_kspace'],output_image=params['output_image'],
    intermediate_output=params['intermediate_output'],batch_size=params['batch_size'],
    mask=params['mask_kspace'],ncoil=params['n_coil'])

# train_gen = generator_type(train_path, batch_size=params['batch_size'])
# val_gen = generator_type(val_path, batch_size=params['batch_size'])

# from generatorKspaceUnet3 import DataGenerator
# train_gen = DataGenerator([r'D:\LB-NN-MONICA-KSPACE\Train-k-space-monica-15.hdf5'],batch_size=batch_size, dim=(256,256)) #loic
# val_gen = DataGenerator([r'D:\LB-NN-MONICA-KSPACE\Val-k-space-monica-15.hdf5'],batch_size=batch_size, dim=(256,256))


params['train_dir'] = os.path.join(params['datadir'],'trainingsaves_{}_{}'.format(agenttag,timestamp))
print('\nusing training directory :\n{}\n'.format(params['train_dir']))
if not os.path.isdir(params['train_dir']):
    os.mkdir(params['train_dir'])
else:
    raise OSError('Train dir already exists!')

print('Saving model and parameters')
model_json = model.to_json()
with open(os.path.join(params['train_dir'],'model_save.json'),'w') as json_file:
    json_file.write(model_json)

with open(os.path.join(params['train_dir'],'params_save.pck'),'wb') as json_file:
    pickle.dump(params,json_file)


tensorboard_callback = TensorBoard(log_dir=os.path.join(params['train_dir'],'log'))
save_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(params['train_dir'],'epoch{epoch:02d}.h5'), save_freq='epoch', save_best_only=False,save_weights_only=True)
save_best_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(params['train_dir'],'best.h5'), save_freq='epoch', save_best_only=True,save_weights_only=True)

model.summary()

print('Training')
history = model.fit(train_gen, validation_data=val_gen, epochs=params['epochs'], callbacks=[tensorboard_callback,save_callback,save_best_callback])#

###evaluation


###plotting
model_path = params['train_dir']
data_file_path = '\\'.join(model_path.split('\\')[:-1]+['val','file1000000.h5'])
data_files = [data_file_path]

extra_plots = ['kspaces','kspaces_diff','mask']

model_output_plotting(data_files,model,model_path,
    plot=False,rewrite=True, only_best=True, 
    intermediate_output=params['intermediate_output'], 
    display_real_imag=params['realimag_img'], input_image=not params['input_kspace'],
    ifft_output=not params['output_image'], 
    ir_kspace=params['realimag_kspace'],ir_img=params['realimag_img'],mask_kspace=params['mask_kspace'],title=params['name'],
    reduced_mask_input=False,extra_plots=extra_plots)

# def write_log(callback, names, logs, batch_no,logDir):
#     writer = tf.summary.create_file_writer(logDir)
#     for name, value in zip(names, logs):
#         with writer.as_default():
#             tf.summary.scalar(name, value, step=batch_no)
#             writer.flush()
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