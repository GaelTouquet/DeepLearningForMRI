from NN.Agent import Agent
from NN.Inputs import prepare_datasets
from NN.Generators import DataGenerator_kspace_img, DataGenerator_img, DataGenerator_kspace, DataGenerator_kspace_to_img, DataGenerator_img_abs
from NN.Masks import RandomMask, CenteredRandomMask, PolynomialMaskGenerator
from NN.architectures import get_wnet, get_unet, nrmse, nrmse_2D_L2, nrmse_2D_L1, testmodel, get_unet_fft, norm_abs, unnorm_abs, simple_dense, reconGAN, double_reconGAN
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
from progress.bar import IncrementalBar

tf.config.set_visible_devices([],'GPU')

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
sampling_factor = 0.15 # fraction of kspace that is measured (=not hidden)
normalise = False# wether to normalize or not, TODO change this to normalisation factor
image_shape = (256,256)
n_channels_in = 1
n_channels_intermediate = 1
n_channels_out = 1

# network

batch_size = 16
epochs = 10
n_tests = 1
# tags for the run
datatag = 'imgnorm_absphasekspaceimg_midslices_kabsmaxnorm' # if already run with this tag, same data is used (=data will not be re-generated)
agenttag = 'ReconGANWnet_abs&phase'#'ReconGAN_absimg'

def my_ssim(y_true, y_pred):
    #abs
    pred_abs = tf.expand_dims(y_pred[:,:,:,0],axis=3)
    pred_abs = tf.add(pred_abs,tf.constant(1.,dtype=y_true.dtype))
    true_abs = tf.expand_dims(y_true[:,:,:,0],axis=3)
    true_abs = tf.add(true_abs,tf.constant(1.,dtype=y_true.dtype))
    ssim_abs = tf.image.ssim(true_abs,pred_abs,2.)
    #phase
    pred_phase = tf.expand_dims(y_pred[:,:,:,1],axis=3)
    pred_phase = tf.add(pred_phase,tf.constant(np.pi,dtype=y_true.dtype))
    true_phase = tf.expand_dims(y_true[:,:,:,1],axis=3)
    true_phase = tf.add(true_phase,tf.constant(np.pi,dtype=y_true.dtype))
    ssim_phase = tf.image.ssim(true_phase,pred_phase,2*np.pi)
    total_ssim = tf.add(ssim_abs,ssim_phase)
    total_ssim = tf.scalar_mul(-1.,total_ssim)
    total_ssim = tf.add(tf.constant(2.,dtype=y_true.dtype),total_ssim)
    return total_ssim

loss='mae'

optimizer='adam'

model = reconGAN((256,256,2),8,depth=3,skip=False)
model.compile(loss=loss,optimizer=optimizer,metrics=['mse','mae'])

indim = image_shape
outdim = image_shape


model_list = {
    'ReconGANUnet_abs&phase_skip_mse' : model,
    # 'ReconGANWnet_abs&phase_skip' : double_reconGAN((256,256,2),16,skip=True),
    # 'ReconGANWnet_abs&phase_noskip' : double_reconGAN((256,256,2),16,skip=False),
    # 'ReconGANUnet_abs&phase_ssim_16filters_4depth_noskip' : reconGAN((256,256,2),16,skip=False),
    # 'ReconGANUnet_abs&phase_ssim_16filters_4depth_skip' : reconGAN((256,256,2),16,skip=True),
    # 'ReconGANUnet_abs&phase_ssim_8filters_4depth_noskip' : reconGAN((256,256,2),16,skip=False),
    # 'ReconGANUnet_abs&phase_ssim_8filters_4depth_skip' : reconGAN((256,256,2),16,skip=True),
    # 'ReconGANUnet_abs&phase_ssim_32filters_4depth_noskip' : reconGAN((256,256,2),16,skip=False),
    # 'ReconGANUnet_abs&phase_ssim_32filters_4depth_skip' : reconGAN((256,256,2),16,skip=True),
    # 'ReconGANUnet_abs&phase_ssim_16filters_3depth_noskip' : reconGAN((256,256,2),16,skip=False,depth=3),
    # 'ReconGANUnet_abs&phase_ssim_16filters_3depth_skip' : reconGAN((256,256,2),16,skip=True,depth=3),
    # 'ReconGANUnet_abs&phase_ssim_8filters_3depth_noskip' : reconGAN((256,256,2),16,skip=False,depth=3),
    # 'ReconGANUnet_abs&phase_ssim_8filters_3depth_skip' : reconGAN((256,256,2),16,skip=True,depth=3),
    # 'ReconGANUnet_abs&phase_ssim_32filters_3depth_noskip' : reconGAN((256,256,2),16,skip=False,depth=3),
    # 'ReconGANUnet_abs&phase_ssim_32filters_3depth_skip' : reconGAN((256,256,2),16,skip=True,depth=3),
    # 'ReconGANUnet_abs&phase_ssim_16filters_5depth_noskip' : reconGAN((256,256,2),16,skip=False,depth=5),
    # 'ReconGANUnet_abs&phase_ssim_16filters_5depth_skip' : reconGAN((256,256,2),16,skip=True,depth=5),
    # 'ReconGANUnet_abs&phase_ssim_8filters_5depth_noskip' : reconGAN((256,256,2),16,skip=False,depth=5),
    # 'ReconGANUnet_abs&phase_ssim_8filters_5depth_skip' : reconGAN((256,256,2),16,skip=True,depth=5),
    # 'ReconGANUnet_abs&phase_ssim_32filters_5depth_noskip' : reconGAN((256,256,2),16,skip=False,depth=5),
    # 'ReconGANUnet_abs&phase_ssim_32filters_5depth_skip' : reconGAN((256,256,2),16,skip=True,depth=5),
}

input_mask = PolynomialMaskGenerator(image_shape,sampling_factor=sampling_factor)#CenteredRandomMask(acceleration=acceleration, center_fraction=(24/360), seed=0xdeadbeef)
for agenttag, model in model_list.items():
    for i_test in range(n_tests):
        # model = reconGAN((256,256,2),16,skip=False)#get_unet(input_shape=(*indim,n_channels_in),depth=8,n_filters=4,batchnorm=False)#get_unet_fft(input_shape=(*indim,n_channels_in),fullskip=True,normfunction=norm_abs, unormfunc=unnorm_abs,depth=6,kernel=2,n_filters=16,batchnorm=False,dropout=0)#get_wnet(input_shape=(*indim,n_channels_in),kernel_initializer=RandomNormal())#not_complex#complex_but_not
        # model = testmodel()
        # json_file = open(r'D:\NN_DATA\singlecoil_acc15_absphasekspaceimg_midslices_kabsmaxnorm\trainingsaves_Unet_kspacetoimg_absnormunnorm_depth9kernel2_nobatchnorm_Feb_09_10_55\model_save.json', 'r')
        # model = model_from_json(json_file.read())
        # json_file.close()
        # model.load_weights(r'D:\NN_DATA\singlecoil_acc15_absphasekspaceimg_midslices_kabsmaxnorm\trainingsaves_Unet_kspacetoimg_absnormunnorm_depth9kernel2_nobatchnorm_Feb_09_10_55\best.h5')



        # model.compile(loss = loss,optimizer=optimizer)#,tf.keras.losses.MeanAbsoluteError() nrmse
        # model.load_weights(r'D:\NN_DATA\singlecoil_acc15_absphasekspaceimg_midslices_kabsmaxnorm\trainingsaves_Unet_img_absnormunnorm_depth8_nobatchnorm_Feb_05_16_26\epoch50.h5')


        
        # model.load_weights(r'D:\NN_DATA\singlecoil_acc15_imgnorm_absphasekspaceimg_midslices_kabsmaxnorm\trainingsaves_ReconGAN_absimg_Feb_12_15_06\best.h5')


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
        train_gen = DataGenerator_img(train_path, batch_size=batch_size)
        val_gen = DataGenerator_img(val_path, batch_size=batch_size)


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
        # print('Training')
        # for i, epoch in enumerate(range(epochs)):
        #     # bar = IncrementalBar('Batch',max=len(train_gen))
        #     for k,batch in enumerate(train_gen):
        #         logs = model.train_on_batch(batch[0],batch[1])
        #         print(logs)
        #         if any([np.any(np.isnan(layer.numpy())) for layer in model.weights]) or logs>1:
        #             print('nan from here')
        #             import pdb;pdb.set_trace()
        #             first=False
        #         write_log(tensorboard_callback,['train_loss'],[logs],i,os.path.join(train_dir,'log'))
        #         model.save_weights((os.path.join(train_dir,'epoch{epoch:02d}_batch{batch:02d}.h5'.format(epoch=i,batch=k))))
        #         last20_weights.append([layer.numpy() for layer in model.weights])
        #         last20_batches.append([batch[0],batch[1]])
        #         last20_loss.append(logs)
                # bar.next()

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