import os
import tensorflow as tf
import pickle
import json
from tensorflow.keras.models import model_from_json
from NN.Generators import DataGenerator
from NN.architectures_cplx import *

def reduced_nrmse(y_true, y_pred):
    denom = tf.keras.backend.mean(tf.sqrt(tf.square(y_true)), axis=(1,2,3))
    nrmse= tf.keras.backend.mean(tf.sqrt(tf.square(y_pred - y_true)), axis=(1,2,3))/denom
    return tf.reduce_mean(nrmse)

def my_ssim(y_true, y_pred):
    tmp_pred = tf.transpose(y_pred, perm=[0,3,1,2,4])
    tmp_true = tf.transpose(y_true, perm=[0,3,1,2,4])
    ssim = tf.image.ssim(tmp_pred,tmp_true,2)
    ssim = tf.math.reduce_mean(ssim,axis=[0,1])
    return 1. - ssim



def evaluate(model, generator, metrics_dict={'mse':'mse'}):
    if metrics_dict:
        metrics = [metrics_dict[name] for name in metrics_dict]
        model.compile(metrics=metrics,loss='mae')
    metrics_list = model.evaluate(generator)
    output_dict = {}
    for metric, value in zip(metrics_dict.keys(),metrics_list):
        output_dict[metric] = value
    return output_dict


if __name__ == '__main__':
    # tf.config.set_visible_devices([],'GPU')
    to_evaluate = {
        'Unet_img_mae' : r'D:\NN_DATA\singlecoil_acc15_ksri_imgri_10midslices_densedpointmasked_imgnorm\trainingsaves_ReconGAN_Unet_img_to_img__nrmse_complex_Apr_08_08_27',
        'Unet_kspace_mae' : r'D:\NN_DATA\singlecoil_acc15_ksri_imgri_10midslices_densedpointmasked_kspace_mask\trainingsaves_ReconGAN_Unet_kspace_to_img_intermoutput_nrmse_complex_Apr_09_19_59',
        'Wnet_mask_mae' : r'D:\NN_DATA\singlecoil_acc15_ksri_imgri_10midslices_densedpointmasked_kspace_mask\trainingsaves_ReconGAN_Unet_kspace_to_img_intermoutput_nrmse_complex_Apr_12_11_55',
    }

    metrics_dict = {
        'mae' : 'mae',
        'mse' : 'mse',
        'ssim' : my_ssim,
    }

    table = {}
    
    for name, path in to_evaluate.items():
        json_file = os.path.join(path,'model_save.json')
        json_file = open(json_file, 'r')
        model = model_from_json(json_file.read())
        json_file.close()
        params = pickle.load(open(os.path.join(path,'params_save.pck'),'rb'))
        input_shape = (*params['image_shape'],2,1) if params['is_complex'] else (*params['image_shape'],2)
        generator = DataGenerator(os.path.join(os.path.split(path)[0],'val'), 
            input_shape, input_kspace=params['input_kspace'],output_image=params['output_image'],
            intermediate_output=params['intermediate_output'],batch_size=params['batch_size'], fraction=params['dataset_fraction'],
            mask=params['mask_kspace'] if 'mask_kspace' in params else False)
        metric_dict = evaluate(model,generator,metrics_dict=metrics_dict)
        out_file = open(os.path.join(path,"eval.json"), "w")
        for mname in metric_dict:
            metric_dict[mname] = '{0:.4f}'.format(metric_dict[mname])
        json.dump(metric_dict,out_file)
        table[name] = metric_dict
        out_file.close()

    for name in table:
        print(name)
        for mname in table[name]:
            print(' '+mname)
            print('  '+table[name][mname]+'\n')