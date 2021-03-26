import os
from NN.Generators import DataGenerator_img,DataGenerator_kspace
import json
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
# tf.config.set_visible_devices([],'GPU')

basedir = r'D:\NN_DATA\singlecoil_acc15_absphasekspaceimg_midslices_kabsmaxnorm'

compute = True

    
requirements = [
    'ReconGANUnet_abs&phase_ssim'
]
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

loss = my_ssim

metrics = [
    'mae',
    'mse'
]

if compute :


    data_files = [f for f in os.listdir(basedir) if (os.path.isdir(os.path.join(basedir, f)) and all([req in f for req in requirements]))]

    info_dict = {}

    for k, f in enumerate(data_files):
        name = f[14:-13]
        print('{}/{} {}'.format(k,len(data_files),name))
        path = os.path.join(basedir,f)
        model_json_file = os.path.join(path,'model_save.json')
        model_weight_file = os.path.join(path,'best.h5')
        info_dict[name] = {
            'name' : name,
            'path' : path,
            'model_json_file' : model_json_file,
            'model_weight_file' : model_weight_file,
            'skip' : 'noskip' not in name,
            'generator_type' : 'kspace',
            'loss_metric' : loss,
        }
        model_json_file = open(model_json_file, 'r')
        model = model_from_json(model_json_file.read())
        model_json_file.close()
        model.compile('adam',loss,metrics)
        model.load_weights(model_weight_file)
        val_gen = DataGenerator_kspace(os.path.join(basedir,'val'),batch_size=10)
        evaluation = model.evaluate(val_gen)
        info_dict[name]['loss'] = evaluation[0]
        for i, metric in enumerate(metrics):
            info_dict[name][metric] = evaluation[i+1]
    with open(os.path.join(basedir,'_'.join(requirements))+'.pkl','wb') as pickle_file:
        pickle.dump(info_dict,pickle_file)
else:
    with open(os.path.join(basedir,'_'.join(requirements))+'.pkl','rb') as pickle_file:
        info_dict = pickle.load(pickle_file)


archs_name = [
    ['ReconGANUnet_abs&phase_ssim_8filters_3depth_noskip','ReconGANUnet_abs&phase_ssim_8filters_4depth_noskip','ReconGANUnet_abs&phase_ssim_8filters_5depth_noskip'],
    ['ReconGANUnet_abs&phase_ssim_16filters_3depth_noskip','ReconGANUnet_abs&phase_ssim_16filters_4depth_noskip','ReconGANUnet_abs&phase_ssim_16filters_5depth_noskip'],
    ['ReconGANUnet_abs&phase_ssim_32filters_3depth_noskip','ReconGANUnet_abs&phase_ssim_32filters_4depth_noskip','ReconGANUnet_abs&phase_ssim_32filters_5depth_noskip'],
    ['ReconGANUnet_abs&phase_ssim_8filters_3depth_skip','ReconGANUnet_abs&phase_ssim_8filters_4depth_skip','ReconGANUnet_abs&phase_ssim_8filters_5depth_skip'],
    ['ReconGANUnet_abs&phase_ssim_16filters_3depth_skip','ReconGANUnet_abs&phase_ssim_16filters_4depth_skip','ReconGANUnet_abs&phase_ssim_16filters_5depth_skip'],
    ['ReconGANUnet_abs&phase_ssim_32filters_3depth_skip','ReconGANUnet_abs&phase_ssim_16filters_3depth_skip','ReconGANUnet_abs&phase_ssim_32filters_5depth_skip']
]

texts = []
colours = []
collabels = ['depth=3','depth=4','depth=5']
rowlabels = ['noskip 8 filters','noskip 16 filters','noskip 32 filters','skip 8 filters','skip 16 filters','skip 32 filters']

for row in archs_name:
    row_text = []
    row_cols = []
    for name in row:
        loss = '{loss:.4f}'.format(loss=info_dict[name]['loss'])
        mae = '{loss:.4f}'.format(loss=info_dict[name]['mae'])
        mse = '{loss:.4f}'.format(loss=info_dict[name]['mse'])
        row_text.append('\n'.join([loss,mse,mae]))
        row_cols.append(info_dict[name]['loss'])
    texts.append(row_text)
    colours.append(row_cols)

fig,ax = plt.subplots()
im = ax.imshow(colours)

for i, row in enumerate(archs_name):
    for k, name in enumerate(row):
        ax.text(k,i,texts[i][k],ha='center',va='center',color='w')

# We want to show all ticks...
ax.set_xticks(np.arange(3))
ax.set_yticks(np.arange(6))
# ... and label them with the respective list entries
ax.set_xticklabels(collabels)
ax.set_yticklabels(rowlabels)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
# fig.tight_layout()
plt.show()
# plt.table(
#     cellText=texts,
#     cellColours=colours,
#     rowLabels=rowlabels,
#     colLabels=collabels
# )