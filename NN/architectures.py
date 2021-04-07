from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Conv2DTranspose, Dropout, Input, Conv2D, Lambda,MaxPooling2D, concatenate, UpSampling2D,Add, BatchNormalization, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np


def nrmse_2D_L1(y_true, y_pred):
    denom1 = tf.keras.backend.mean(tf.abs(y_true[:,:,:,0]), axis=(1,2))
    num1 = tf.keras.backend.mean(tf.abs(y_pred[:,:,:,0] - y_true[:,:,:,0]), axis=(1,2))
    denom2 = tf.keras.backend.mean(tf.abs(y_true[:,:,:,1]), axis=(1,2))
    num2 = tf.keras.backend.mean(tf.abs(y_pred[:,:,:,1] - y_true[:,:,:,1]), axis=(1,2))
    return (num1/denom1) + (num2/denom2)

def nrmse_2D_L2(y_true, y_pred):
    denom1 = tf.sqrt(tf.keras.backend.mean(tf.square(y_true[:,:,:,0]), axis=(1,2)))
    num1 = tf.sqrt(tf.keras.backend.mean(tf.square(y_pred[:,:,:,0] - y_true[:,:,:,0]), axis=(1,2)))
    denom2 = tf.sqrt(tf.keras.backend.mean(tf.square(y_true[:,:,:,1]), axis=(1,2)))
    num2 = tf.sqrt(tf.keras.backend.mean(tf.square(y_pred[:,:,:,1] - y_true[:,:,:,1]), axis=(1,2)))
    return (num1/denom1) + (num2/denom2)

def nrmse(y_true, y_pred):
    denom = tf.sqrt(tf.keras.backend.mean(tf.square(y_true), axis=(1,2,3)))
    return tf.sqrt(tf.keras.backend.mean(tf.square(y_pred - y_true), axis=(1,2,3)))\
    /denom

def reduced_nrmse(y_true, y_pred):
    denom = tf.sqrt(tf.keras.backend.mean(tf.square(y_true), axis=(1,2,3)))
    nrmse= tf.sqrt(tf.keras.backend.mean(tf.square(y_pred - y_true), axis=(1,2,3)))/denom
    return tf.reduce_mean(nrmse)

def my_ssim(y_true, y_pred):
    # pred_abs = tf.math.subtract(y_pred[:,:,:,0], tf.math.reduce_min(y_pred[:,:,:,0],axis=[1,2]))
    # pred_abs = tf.scalar_mul(tf.math.reduce_max(y_pred[:,:,:,0],axis=[1,2]),y_pred[:,:,:,0])
    # pred_abs = tf.math.subtract( tf.constant(1.,dtype=pred_abs.dtype), tf.math.scalar_mul(2.,pred_abs))
    #abs
    pred_abs = tf.expand_dims(y_pred[:,:,:,0],axis=3)
    pred_abs = tf.add(pred_abs,tf.constant(1.,dtype=y_true.dtype))
    true_abs = tf.expand_dims(y_true[:,:,:,0],axis=3)
    true_abs = tf.add(true_abs,tf.constant(1.,dtype=y_true.dtype))
    ssim_abs = tf.image.ssim(true_abs,pred_abs,2.)
    #phase #Loic decomment following
    # pred_phase = tf.expand_dims(y_pred[:,:,:,1],axis=3)
    # pred_phase = tf.add(pred_phase,tf.constant(np.pi,dtype=y_true.dtype))
    # true_phase = tf.expand_dims(y_true[:,:,:,1],axis=3)
    # true_phase = tf.add(true_phase,tf.constant(np.pi,dtype=y_true.dtype))
    # ssim_phase = tf.image.ssim(true_phase,pred_phase,2*np.pi)
    #combination
    # total_ssim = tf.add(ssim_abs,ssim_phase)
    total_ssim = ssim_abs #loic remove this
    total_ssim = tf.scalar_mul(-1.,total_ssim)
    total_ssim = tf.add(tf.constant(2.,dtype=y_true.dtype),total_ssim)
    return total_ssim


def norm_abs(image):
    k_abs = image[:,:,:,0]
    k_angle = image[:,:,:,1]
    return tf.stack([1e5*k_abs,k_angle],axis=-1)

def unnorm_abs(image):
    k_abs = image[:,:,:,0]
    k_angle = image[:,:,:,1]
    return tf.stack([k_abs/1e5,k_angle],axis=-1)

# class IfftLayer(tf.keras.layers.Layer):
#     def call(self, kspace):
#         k_abs = kspace[:,:,:,0]
#         k_angle = kspace[:,:,:,1]
#         k_angle 
#         real = k_abs * tf.math.cos(k_angle)
#         imag = k_abs * tf.math.sin(k_angle)
#         kspace_complex = tf.complex(real,imag)
#         kspace_complex = tf.signal.ifftshift(kspace_complex,axes=[1,2])
#         rec1 = tf.signal.ifft2d(kspace_complex)
#         rec1 = tf.signal.fftshift(rec1,axes=[1,2])
#         the_abs = tf.math.abs(rec1)
#         factors = tf.math.reduce_max(the_abs,axis=[1,2])
#         the_abs = tf.math.divide(the_abs,tf.expand_dims(tf.expand_dims(factors,axis=1),axis=2))
#         the_abs = tf.math.scalar_mul(2.,the_abs)
#         the_abs = the_abs - 1.
#         rec1 = [the_abs,tf.math.angle(rec1)]
#         return tf.stack(rec1,axis=-1)

def ifft_layer(realimag_kspace,realimag_img,img_norm):
    def actual_ifft_layer_func(kspace):

        k_0 = kspace[:,:,:,0]
        k_1 = kspace[:,:,:,1]

        kspace_complex = tf.complex(k_0,k_1)
        kspace_complex = tf.signal.ifftshift(kspace_complex,axes=[1,2])
        image_complex = tf.signal.ifft2d(kspace_complex)
        image_complex = tf.signal.fftshift(image_complex,axes=[1,2])
        
        if img_norm['tf']:
            image_complex = img_norm['tf'](image_complex)
        # return tf.expand_dims(tf.abs(image_complex), -1)
        #loic decomment following and remove return
        if realimag_img:
            i_0 = tf.math.real(image_complex)
            i_1 = tf.math.imag(image_complex)
        else:
            i_0 = tf.math.abs(image_complex)
            i_1 = tf.math.angle(image_complex)
        #     if normalise_image:
        #         i_0_norm_factors = tf.math.reduce_max(i_0,axis=[1,2])
        #         i_0 = tf.math.divide(i_0, tf.expand_dims(tf.expand_dims(i_0_norm_factors,axis=1),axis=2))
        #         if center_normalised_values:
        #             i_0 = tf.math.scalar_mul(2.,i_0) - 1.
        return tf.stack([i_0,i_1],axis=3)
    return actual_ifft_layer_func

def dense_block(input_tensor,dense_shape):
    x = input_tensor
    for n_neurons in dense_shape:
        x = Dense(n_neurons)(x)
    return x

def simple_dense(dense_shape=[],input_shape=(256,256,2),do_ifft=False):
    inputs = Input(shape=input_shape)
    x = dense_block(inputs,dense_shape)
    x = Dense(256*256*2)(x)
    x = tf.reshape(x,(256,256,2))

    if do_ifft:
        x = Lambda(ifft_layer)(x)

    model = Model(inputs=inputs,outputs=x)
    return model

### ReconGAN blocks

def encoder_block(input_tensor,n_filters,kernel_initializer='glorot_uniform',dropout=False):
    conv_a = Conv2D(n_filters,(3,3),strides=2,padding='same',kernel_initializer=kernel_initializer)(input_tensor)
    conv_i = Conv2D(n_filters,(3,3),padding='same',kernel_initializer=kernel_initializer)(conv_a)
    conv_m = Conv2D(int(n_filters/2),(3,3),padding='same',kernel_initializer=kernel_initializer)(conv_i)
    conv_o = Conv2D(n_filters,(3,3),padding='same',kernel_initializer=kernel_initializer)(conv_m)
    conc_o = Add()([conv_o,conv_a])
    conv_b = Conv2D(n_filters,(3,3),padding='same',kernel_initializer=kernel_initializer)(conc_o)
    if dropout:
        conv_b = Dropout(dropout)(conv_b)
    return conv_b

def decoder_block(input_tensor,n_filters,kernel_initializer='glorot_uniform',dropout=False):
    conv_a = Conv2DTranspose(n_filters,(3,3),padding='same',kernel_initializer=kernel_initializer)(input_tensor)
    conv_i = Conv2D(n_filters,(3,3),padding='same',kernel_initializer=kernel_initializer)(conv_a)
    conv_m = Conv2D(int(n_filters/2),(3,3),padding='same',kernel_initializer=kernel_initializer)(conv_i)
    conv_o = Conv2D(n_filters,(3,3),padding='same',kernel_initializer=kernel_initializer)(conv_m)
    conc_o = Add()([conv_o,conv_a])
    conv_b = Conv2DTranspose(n_filters,(3,3),strides=2,padding='same',kernel_initializer=kernel_initializer)(conc_o)
    if dropout:
        conv_b = Dropout(dropout)(conv_b)
    return conv_b


def reconGAN_Unet_block(input_tensor,n_filters,depth=4,skip=False,kernel_initializer='glorot_uniform',dropout=False):

    downstream = []
    x = input_tensor

    for i in range(depth):
        x = encoder_block(x,n_filters*(2**i),kernel_initializer=kernel_initializer,dropout=dropout)
        downstream.append(x)

    for i in range(depth):
        x = decoder_block(x,n_filters*(2**(depth-(i+1))),kernel_initializer=kernel_initializer,dropout=dropout)
        if i==depth-1:
            continue
        x = concatenate([x, downstream[-1*(i+2)]])

    x = Conv2D(2, (1, 1), activation='linear') (x)
    # #loic uncomment following and remove previous
    # x = Conv2D(input_tensor.shape[-1],(1,1),padding='same',activation='tanh',kernel_initializer=kernel_initializer)(x)

    # x = Lambda(lambda t: tf.stack([t[:,:,:,0],tf.scalar_mul(np.pi,t[:,:,:,1])],axis=-1))(x)

    if dropout:
        x = Dropout(dropout)(x)

    if skip:
        x = Add()([input_tensor,x])

    return x

### ReconGAN Unets

def reconGAN_Unet(input_shape,n_filters,depth=4,skip=False,kernel_initializer='glorot_uniform',dropout=False):
    inputs = Input(shape=input_shape)

    outputs = reconGAN_Unet_block(inputs,n_filters,depth,skip=skip,kernel_initializer=kernel_initializer,dropout=dropout)
    
    return Model(inputs=inputs,outputs=outputs)

def reconGAN_Wnet(input_shape,n_filters_kspace,n_filters_img,depth=4,skip=False,kernel_initializer='glorot_uniform',dropout=False,
realimag_kspace=True,realimag_img=True,normalise_image=True,center_normalised_values=True):
    inputs = Input(shape=input_shape)

    kspace_out = reconGAN_Unet_block(inputs,n_filters_kspace,depth,skip=skip,kernel_initializer=kernel_initializer,dropout=dropout)

    image_in = Lambda(ifft_layer(realimag_kspace,realimag_img,normalise_image,center_normalised_values))(kspace_out)

    image_out = reconGAN_Unet_block(image_in,n_filters_img,depth,skip=skip,kernel_initializer=kernel_initializer,dropout=dropout)

    return Model(inputs=inputs,outputs=image_out)

def reconGAN_Wnet_intermediate(input_shape,n_filters_kspace,n_filters_img,depth=4,skip=False,kernel_initializer='glorot_uniform',dropout=False,
realimag_kspace=True,realimag_img=True,normalise_image=True,center_normalised_values=True):
    inputs = Input(shape=input_shape)

    kspace_out = reconGAN_Unet_block(inputs,n_filters_kspace,depth,skip=skip,kernel_initializer=kernel_initializer,dropout=dropout)

    image_in = Lambda(ifft_layer(realimag_kspace,realimag_img,normalise_image=normalise_image,center_normalised_values=center_normalised_values))(kspace_out)#loic: 

    image_out = reconGAN_Unet_block(image_in,n_filters_img,depth,skip=skip,kernel_initializer=kernel_initializer,dropout=dropout)#loic skip=skip

    return Model(inputs=inputs,outputs=[kspace_out,image_out])#loic : image_in instead of kspace_out

    
def reconGAN_Unet_kspace_to_img(input_shape,n_filters_kspace,depth=4,skip=False,kernel_initializer='glorot_uniform',dropout=False,
realimag_kspace=True,realimag_img=True,img_norm=None):
    inputs = Input(shape=input_shape)

    kspace_out = reconGAN_Unet_block(inputs,n_filters_kspace,depth,skip=skip,kernel_initializer=kernel_initializer,dropout=dropout)

    image_in = Lambda(ifft_layer(realimag_kspace,realimag_img,img_norm))(kspace_out)

    return Model(inputs=inputs,outputs=[kspace_out,image_in])

# def find_architecture(arch_type,input_kspace,output_image,intermediate_output=False):
#     if arch_type=='ReconGAN_Unet':
#         if input_kspace:
#             if output_image:
#                 return reconGAN_Unet_kspace_to_img
#             else:
#                 return reconGAN_Unet
#         else:
#             if output_image:
#                 return reconGAN_Unet
#             else:
#                 raise ValueError('Why would you want to have image input and kspace output?!')
#     elif arch_type=='ReconGAN_Wnet':
#         if (not input_kspace) or (not output_image):
#             raise ValueError('Wnets are meant to have kspace input and image output!')
#         if intermediate_output:
#             return reconGAN_Wnet_intermediate
#         else:
#             return reconGAN_Wnet
#     else:
#         raise ValueError('architecture type {} not recognized/implemented'.format(arch_type))

#### RNNs

class CRNNi(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, padding, activation,**kwargs):
        super(CRNNi, self).__init__(**kwargs)
        self.global_iterations = Conv2D(filters=filters,kernel_size=kernel_size,padding=padding, use_bias=False)
        self.from_temporal_iterations = Conv2D(filters=filters,kernel_size=kernel_size,padding=padding, use_bias=False)
        self.activation = activation

    def call(self, gi_inputs, ti_inputs):
        gi = self.global_iterations(gi_inputs)
        ti = self.from_temporal_iterations(ti_inputs)
        return self.activation(tf.add(gi,ti))

class BCRNN(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, padding, activation,**kwargs):
        super(BCRNN, self).__init__(**kwargs)
        self.temporal = Conv2D(filters=filters,kernel_size=kernel_size,padding=padding, use_bias=False)
        self.global_iterations = Conv2D(filters=filters,kernel_size=kernel_size,padding=padding, use_bias=False)