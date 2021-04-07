from tensorflow.keras.layers import Activation, Conv2DTranspose, Dropout, Input, Conv2D, Lambda,MaxPooling2D, concatenate, UpSampling2D,Add, BatchNormalization, Dense, Bidirectional
from tensorflow.keras.models import Model
import tensorflow as tf

def reduced_nrmse(y_true, y_pred):
    denom = tf.keras.backend.mean(tf.sqrt(tf.square(y_true)), axis=(1,2,3))
    nrmse= tf.keras.backend.mean(tf.sqrt(tf.square(y_pred - y_true)), axis=(1,2,3))/denom
    return tf.reduce_mean(nrmse)

def my_ssim(y_true, y_pred):
    # pred_abs = tf.math.subtract(y_pred[:,:,:,0], tf.math.reduce_min(y_pred[:,:,:,0],axis=[1,2]))
    # pred_abs = tf.scalar_mul(tf.math.reduce_max(y_pred[:,:,:,0],axis=[1,2]),y_pred[:,:,:,0])
    # pred_abs = tf.math.subtract( tf.constant(1.,dtype=pred_abs.dtype), tf.math.scalar_mul(2.,pred_abs))
    #abs
    pred_abs = tf.expand_dims(y_pred[:,:,:,0,0],axis=3)
    # pred_abs = tf.add(pred_abs,tf.constant(1.,dtype=y_true.dtype))
    true_abs = tf.expand_dims(y_true[:,:,:,0,0],axis=3)
    # true_abs = tf.add(true_abs,tf.constant(1.,dtype=y_true.dtype))
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

def conv2d_cplx(input_tf, num_features, kernel_size, strides=1, dilation_rate=(1, 1), use_bias=True, kernel_initializer='glorot_uniform',padding='same'):

    real_part = input_tf[:,:,:,0,:]
    imag_part = input_tf[:,:,:,1,:]

    realconv = Conv2D(num_features,kernel_size=kernel_size,strides=strides,dilation_rate=dilation_rate,use_bias=use_bias,kernel_initializer=kernel_initializer,padding=padding)
    imagconv = Conv2D(num_features,kernel_size=kernel_size,strides=strides,dilation_rate=dilation_rate,use_bias=use_bias,kernel_initializer=kernel_initializer,padding=padding)

    real_to_real = realconv(real_part)
    real_to_imag = imagconv(real_part)
    imag_to_imag = realconv(imag_part)
    imag_to_real = imagconv(imag_part)

    realout = real_to_real - imag_to_real
    imagout = real_to_imag + imag_to_imag

    output_tf = tf.stack([realout,imagout],axis=3)

    return output_tf

def conv2d_transpose_cplx(input_tf, num_features, kernel_size, strides=1, dilation_rate=(1, 1), use_bias=True, kernel_initializer='glorot_uniform',padding='same'):

    real_part = input_tf[:,:,:,0,:]
    imag_part = input_tf[:,:,:,1,:]

    #Just real conv
    real_to_real_conv = Conv2DTranspose(num_features,kernel_size=kernel_size,strides=strides,dilation_rate=dilation_rate,use_bias=use_bias,kernel_initializer=kernel_initializer,padding=padding)
    imag_to_imag_conv = Conv2DTranspose(num_features,kernel_size=kernel_size,strides=strides,dilation_rate=dilation_rate,use_bias=use_bias,kernel_initializer=kernel_initializer,padding=padding)
 
    realout = real_to_real_conv(real_part)
    imagout = imag_to_imag_conv(imag_part)

    #from article
    # real_to_im_conv = Conv2DTranspose(num_features,kernel_size=kernel_size,strides=strides,dilation_rate=dilation_rate,use_bias=use_bias,kernel_initializer=kernel_initializer,padding=padding)
    # imag_to_im_conv = Conv2DTranspose(num_features,kernel_size=kernel_size,strides=strides,dilation_rate=dilation_rate,use_bias=use_bias,kernel_initializer=kernel_initializer,padding=padding)
    # real_to_re_conv = Conv2DTranspose(num_features,kernel_size=kernel_size,strides=strides,dilation_rate=dilation_rate,use_bias=use_bias,kernel_initializer=kernel_initializer,padding=padding)
    # imag_to_re_conv = Conv2DTranspose(num_features,kernel_size=kernel_size,strides=strides,dilation_rate=dilation_rate,use_bias=use_bias,kernel_initializer=kernel_initializer,padding=padding)

    # real_to_real = real_to_re_conv(real_part)
    # real_to_imag = real_to_im_conv(real_part)
    # imag_to_imag = imag_to_im_conv(imag_part)
    # imag_to_real = imag_to_re_conv(imag_part)

    #from theory
    # real_conv = Conv2DTranspose(num_features,kernel_size=kernel_size,strides=strides,dilation_rate=dilation_rate,use_bias=use_bias,kernel_initializer=kernel_initializer,padding=padding)
    # imag_conv = Conv2DTranspose(num_features,kernel_size=kernel_size,strides=strides,dilation_rate=dilation_rate,use_bias=use_bias,kernel_initializer=kernel_initializer,padding=padding)

    # real_to_real = real_conv(real_part)
    # real_to_imag = imag_conv(real_part)
    # imag_to_imag = real_conv(imag_part)
    # imag_to_real = imag_conv(imag_part)

    # realout = real_to_real - imag_to_real
    # imagout = real_to_imag + imag_to_imag

    output_tf = tf.stack([realout,imagout],axis=3)

    return output_tf

def z_relu(input_tf):

    real_part = input_tf[:,:,:,0,:]
    imag_part = input_tf[:,:,:,1,:]

    is_real_pos = tf.math.less_equal(real_part,0.)
    is_imag_pos = tf.math.less_equal(imag_part,0.)

    zeros = tf.zeros_like(real_part)
    # ones = tf.ones_like(real_part)
    # real_pos = tf.where(is_real_pos,ones,zeros)
    # imag_pos = tf.where(is_imag_pos,ones,zeros)

    # real_part = real_part * real_pos * imag_pos
    # imag_part = imag_part * real_pos * imag_pos*
    real_part = tf.where(is_real_pos,real_part,zeros)
    real_part = tf.where(is_imag_pos,real_part,zeros)
    imag_part = tf.where(is_real_pos,real_part,zeros)
    imag_part = tf.where(is_imag_pos,real_part,zeros)

    output_tf = tf.stack([real_part,imag_part],axis=3)

    return output_tf

### ReconGAN blocks

def encoder_block(input_tensor,n_filters,kernel_size=(3,3),kernel_initializer='glorot_uniform',dropout=False,batchnorm=False, activation=None):
    conv_a = conv2d_cplx(input_tensor,n_filters,kernel_size,strides=2,kernel_initializer=kernel_initializer)
    conv_i = conv2d_cplx(conv_a,n_filters,kernel_size,kernel_initializer=kernel_initializer)
    conv_m = conv2d_cplx(conv_i,int(n_filters/2),kernel_size,kernel_initializer=kernel_initializer)
    conv_o = conv2d_cplx(conv_m,n_filters,kernel_size,kernel_initializer=kernel_initializer)
    conv_o = Add()([conv_o,conv_a])
    conv_b = conv2d_cplx(conv_o,n_filters,kernel_size,kernel_initializer=kernel_initializer)
    if dropout:
        conv_b = Dropout(dropout)(conv_b)
    if batchnorm:
        conv_b = BatchNormalization(conv_b)
    if activation:
        conv_b = activation(conv_b)
    return conv_b

def decoder_block(input_tensor,n_filters,kernel_size=(3,3),kernel_initializer='glorot_uniform',dropout=False,batchnorm=False, activation=None):
    conv_a = conv2d_transpose_cplx(input_tensor,n_filters,kernel_size,strides=2,kernel_initializer=kernel_initializer)
    conv_i = conv2d_cplx(conv_a,n_filters,kernel_size,kernel_initializer=kernel_initializer)
    conv_m = conv2d_cplx(conv_i,int(n_filters/2),kernel_size,kernel_initializer=kernel_initializer)
    conv_o = conv2d_cplx(conv_m,n_filters,kernel_size,kernel_initializer=kernel_initializer)
    conv_o = Add()([conv_o,conv_a])
    conv_b = conv2d_transpose_cplx(conv_o,n_filters,kernel_size,kernel_initializer=kernel_initializer)
    if dropout:
        conv_b = Dropout(dropout)(conv_b)
    if batchnorm:
        conv_b = BatchNormalization(conv_b)
    if activation:
        conv_b = activation(conv_b)
    return conv_b

def reconGAN_Unet_block(input_tensor,n_filters,depth=4,skip=False,kernel_initializer='glorot_uniform',dropout=False, activation=None):

    downstream = []
    x = input_tensor

    for i in range(depth):
        x = encoder_block(x,n_filters*(2**i),kernel_initializer=kernel_initializer,dropout=dropout,activation=activation)
        downstream.append(x)

    for i in range(depth):
        x = decoder_block(x,n_filters*(2**(depth-(i+1))),kernel_initializer=kernel_initializer,dropout=dropout,activation=activation)
        if i==depth-1:
            continue
        x = concatenate([x, downstream[-1*(i+2)]])

    x = conv2d_cplx(x, 1, (1, 1))

    if dropout:
        x = Dropout(dropout)(x)

    if skip:
        x = Add()([input_tensor,x])

    return x

def ifft_layer(realimag_kspace,realimag_img,img_norm):
    def actual_ifft_layer_func(kspace):
        real_part = kspace[:,:,:,0,:]
        imag_part = kspace[:,:,:,1,:]
        kspace_complex = tf.complex(real_part,imag_part)
        kspace_complex = tf.transpose(kspace_complex,perm=[0,3,1,2])
        kspace_complex = tf.signal.ifftshift(kspace_complex,axes=[2,3])
        image_complex = tf.signal.ifft2d(kspace_complex)
        image_complex = tf.signal.fftshift(image_complex,axes=[2,3])
        image_complex = tf.transpose(image_complex,perm=[0,2,3,1])
        if img_norm['tf']:
            image_complex = img_norm['tf'](image_complex)
        return tf.stack([tf.math.real(image_complex),tf.math.imag(image_complex)],axis=3)
    return actual_ifft_layer_func


### ReconGAN Unets

def reconGAN_Unet(input_shape,n_filters,depth=4,skip=False,kernel_initializer='glorot_uniform',dropout=False,activation=z_relu):
    inputs = Input(shape=input_shape)

    outputs = reconGAN_Unet_block(inputs,n_filters,depth,skip=skip,kernel_initializer=kernel_initializer,dropout=dropout,activation=activation)
    
    return Model(inputs=inputs,outputs=outputs)

def reconGAN_Wnet(input_shape,n_filters_kspace,n_filters_img,depth=4,skip=False,kernel_initializer='glorot_uniform',dropout=False,
realimag_kspace=True,realimag_img=True,img_norm=None,activation=z_relu):
    inputs = Input(shape=input_shape)

    kspace_out = reconGAN_Unet_block(inputs,n_filters_kspace,depth,skip=skip,kernel_initializer=kernel_initializer,dropout=dropout,activation=activation)

    image_in = Lambda(ifft_layer(realimag_kspace,realimag_img,img_norm))(kspace_out)

    image_out = reconGAN_Unet_block(image_in,n_filters_img,depth,skip=skip,kernel_initializer=kernel_initializer,dropout=dropout,activation=activation)

    return Model(inputs=inputs,outputs=image_out)

def reconGAN_Wnet_intermediate(input_shape,n_filters_kspace,n_filters_img,depth=4,skip=False,kernel_initializer='glorot_uniform',dropout=False,
realimag_kspace=True,realimag_img=True,img_norm=None,activation=z_relu):
    inputs = Input(shape=input_shape)

    kspace_out = reconGAN_Unet_block(inputs,n_filters_kspace,depth,skip=skip,kernel_initializer=kernel_initializer,dropout=dropout,activation=activation)

    image_in = Lambda(ifft_layer(realimag_kspace,realimag_img,img_norm))(kspace_out)#loic: 

    image_out = reconGAN_Unet_block(image_in,n_filters_img,depth,skip=skip,kernel_initializer=kernel_initializer,dropout=dropout,activation=activation)#loic skip=skip

    return Model(inputs=inputs,outputs=[kspace_out,image_out])#loic : image_in instead of kspace_out

    
def reconGAN_Unet_kspace_to_img(input_shape,n_filters_kspace,depth=4,skip=False,kernel_initializer='glorot_uniform',dropout=False,
realimag_kspace=True,realimag_img=True,img_norm=None,activation=z_relu):
    inputs = Input(shape=input_shape)

    kspace_out = reconGAN_Unet_block(inputs,n_filters_kspace,depth,skip=skip,kernel_initializer=kernel_initializer,dropout=dropout,activation=activation)

    image_in = Lambda(ifft_layer(realimag_kspace,realimag_img,img_norm))(kspace_out)

    return Model(inputs=inputs,outputs=[kspace_out,image_in])