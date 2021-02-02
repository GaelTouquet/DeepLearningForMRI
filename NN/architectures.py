import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Conv2DTranspose, Dropout, Input, Conv2D, Lambda,MaxPooling2D, concatenate, UpSampling2D,Add, BatchNormalization
from tensorflow.keras.optimizers import Adam

def nrmse_2D(y_true, y_pred):
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


def ifft_layer(kspace):
    real = Lambda(lambda kspace : kspace[:,:,:,0])(kspace)
    imag = Lambda(lambda kspace : kspace[:,:,:,1])(kspace)
    kspace_complex = tf.complex(real,imag)
    kspace_complex = tf.signal.ifftshift(kspace_complex,axes=[1,2])
    rec1 = tf.signal.ifft2d(kspace_complex)
    rec1 = tf.signal.fftshift(rec1,axes=[1,2])
    rec1 = [tf.math.abs(rec1),tf.math.angle(rec1)]
    return tf.stack(rec1,axis=-1)# rec1 = tf.expand_dims(rec1, -1)
    # return tf.expand_dims(tf.abs(rec1),3)

def conv2d_block(input_tensor, n_filters, kernel_size=3, n_layer=1, batchnorm=True,kernel_initializer="he_normal"):
    x = input_tensor
    for i in range(n_layer):
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer=kernel_initializer,
            padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x

def conv2d_block_3(input_tensor, n_filters, kernel_size=3, batchnorm=True,kernel_initializer="he_normal"):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer=kernel_initializer,
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer=kernel_initializer,
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # third layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer=kernel_initializer,
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def conv2d_block_2(input_tensor, n_filters, kernel_size=3, batchnorm=True,kernel_initializer="he_normal"):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer=kernel_initializer,
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer=kernel_initializer,
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def test_model(input_shape=(200,200,2,1)):
    inputs = Input(shape=input_shape)
    ifft = Lambda(ifft_layer)(inputs)
    model = Model(inputs=inputs,outputs=ifft)
    return model

def get_unet(input_shape=(200,200,2,1), n_layer_kspace=3, n_layer_img=3, n_filters=16, dropout=0.5, batchnorm=True, kernel_initializer="he_normal", normfactor=1):
    
#    print(input_img)
    # contracting path
#    input_img=(input_img-1000)/1000
    inputs = Input(shape=input_shape)

    #normfactor test
    normed_inputs = Lambda(lambda x: x*normfactor)(inputs)

    c1 = conv2d_block(normed_inputs, n_layer=n_layer_kspace,n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_layer=n_layer_kspace, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_layer=n_layer_kspace, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_layer=n_layer_kspace, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_layer=n_layer_kspace, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_layer=n_layer_kspace, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_layer=n_layer_kspace, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_layer=n_layer_kspace, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_layer=n_layer_kspace, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
   
    outkspace = Conv2D(2, (1, 1), activation='linear') (c9)

    outkspace_comb = Add()([outkspace,normed_inputs])

    #normfactor test
    outkspace_comb = Lambda(lambda x: x/normfactor)(outkspace_comb)

    img_rec = Lambda(ifft_layer)(outkspace_comb)

    #normfactor test
    img_rec_norm = Lambda(lambda x: x*normfactor)(img_rec)
    
    d1 = conv2d_block(img_rec_norm, n_layer=n_layer_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block(img_rec, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    k1 = MaxPooling2D((2, 2)) (d1)
    k1 = Dropout(dropout*0.5)(k1)

    d2 = conv2d_block(k1, n_layer=n_layer_img, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block(k1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    k2 = MaxPooling2D((2, 2)) (d2)
    k2 = Dropout(dropout)(k2)

    d3 = conv2d_block(k2, n_layer=n_layer_img, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block(k2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    k3 = MaxPooling2D((2, 2)) (d3)
    k3 = Dropout(dropout)(k3)

    d4 = conv2d_block(k3, n_layer=n_layer_img, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block(k3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    k4 = MaxPooling2D(pool_size=(2, 2)) (d4)
    k4 = Dropout(dropout)(k4)
    
    d5 = conv2d_block(k4, n_layer=n_layer_img, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block(k4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    
    # expansive path
    v6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (d5)
    v6 = concatenate([v6, d4])
    v6 = Dropout(dropout)(v6)
    d6 = conv2d_block(v6, n_layer=n_layer_img, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block_2(v6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)

    v7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (d6)
    v7 = concatenate([v7, d3])
    v7 = Dropout(dropout)(v7)
    d7 = conv2d_block(v7, n_layer=n_layer_img, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block_2(v7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)

    v8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (d7)
    v8 = concatenate([v8, d2])
    v8 = Dropout(dropout)(v8)
    d8 = conv2d_block(v8, n_layer=n_layer_img, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block_2(v8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)

    v9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (d8)
    v9 = concatenate([v9, d1], axis=3)
    v9 = Dropout(dropout)(v9)
    d9 = conv2d_block(v9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block_2(v9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
   
    outimg = Conv2D(2, (1, 1), activation='linear') (d9)

    outimg_comb = Add()([img_rec_norm,outimg])
    
    #normfactor test
    outimg_comb = Lambda(lambda x: x/normfactor)(outimg_comb)

    model = Model(inputs=[inputs], outputs=[img_rec,outimg_comb])#
    return model

# from tensorflow.keras import backend as keras
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, concatenate, UpSampling2D, Lambda, Add, BatchNormalization
# from tensorflow.keras.models import Model
# import tensorflow as tf
# from utils.fastMRI_utils import crop
# from NN.Layers import ComplexConvAdder

# UnstackLayer = Lambda(lambda x: tf.unstack(x,axis=-2))
# StackLayer = Lambda(lambda x: tf.stack(x,axis=-2))

# def ifft_func(x):
#     """
#     applies ifft to list of [real, cplx] tensors in input
#     """
#     x = tf.unstack(x,axis=-2)
#     x = tf.dtypes.complex(x[0],x[1])
#     x = tf.transpose(x,perm=[0,3,1,2])
#     x = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(x)))
#     x = tf.transpose(x,perm=[0,2,3,1])
#     x = [tf.math.real(x),tf.math.imag(x)]
#     return tf.stack(x,axis=-2)
#     # return [tf.math.real(x),tf.math.imag(x)]

# IfftLayer = Lambda(ifft_func)

# def ifft_func_complexchannels(x):
#     x = tf.unstack(x,axis=-1)
#     x = tf.dtypes.complex(x[0],x[1])
#     x = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(x)))
#     x = [tf.math.real(x),tf.math.imag(x)]
#     return tf.stack(x,axis=-1)

# IfftLayer_complexchannels = Lambda(ifft_func_complexchannels)

# def cplx_conv_adding(x):
#     real = tf.math.subtract(x[0],x[1])
#     cplx = tf.math.add(x[2],x[3])
#     return [real,cplx]

# ComplexConvAdderLayer = Lambda(cplx_conv_adding)

# SubstracterLayer = Lambda(lambda x: tf.math.subtract(x[0],x[1]))

# def my_abs(x):
#     x = tf.unstack(x,axis=-2)
#     x = tf.dtypes.complex(x[0],x[1])
#     return tf.math.abs(x)

# AbsLayer = Lambda(my_abs)


# def complexconv(x,n_conv=2, filters= 128, kernel_size = (5,5), activation='relu'):
#     real,cplx = UnstackLayer(x)
#     for i_conv in range(n_conv):
#         realconv = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding = 'same')
#         cplxconv = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding = 'same')
#         real_from_real = realconv(real)
#         real_from_cplx = cplxconv(cplx)
#         cplx_from_real = cplxconv(real)
#         cplx_from_cplx = realconv(cplx)
#         real, cplx = ComplexConvAdderLayer([real_from_real,real_from_cplx,cplx_from_real,cplx_from_cplx])
#     return StackLayer([real,cplx])

# def complexmaxpool(x,pool_size=(2,2)):
#     real,cplx = UnstackLayer(x)
#     pool = MaxPooling2D(pool_size=(2, 2))
#     real = pool(real)
#     cplx = pool(cplx)
#     return StackLayer([real,cplx])

# def complexupsample(x,size=(2,2)):
#     real,cplx = UnstackLayer(x)
#     up = UpSampling2D(size=(2, 2))
#     real = up(real)
#     cplx = up(cplx)
#     return StackLayer([real,cplx])

# def not_complex(input_shape=(40,40,2)):
#     inputs = Input(shape=input_shape)

#     Block1 = Conv2D(filters=128, kernel_size=(5,5), activation='relu',padding='same')(inputs)
#     Block1 = Conv2D(filters=1, kernel_size=(5,5), activation='relu',padding='same')(Block1)
#     Block1 = BatchNormalization()(Block1)
#     Block1 = Add()([inputs,Block1])

#     Block2 = Conv2D(filters=128, kernel_size=(5,5), activation='relu',padding='same')(inputs)
#     Block2 = Conv2D(filters=1, kernel_size=(5,5), activation='relu',padding='same')(Block2)
#     Block2 = BatchNormalization()(Block2)
#     Block2 = Add()([inputs,Block2])

#     ifft = IfftLayer_complexchannels(Block2)

#     down1 = Conv2D(32,3,activation='relu',padding='same')(ifft)
#     down1 = Conv2D(32,3,activation='relu',padding='same')(down1)
#     down1 = BatchNormalization()(down1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(down1)


#     down2 = Conv2D(64,3,activation='relu',padding='same')(pool1)
#     down2 = Conv2D(64,3,activation='relu',padding='same')(down2)
#     down2 = BatchNormalization()(down2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(down2)
    
#     down3 = Conv2D(128,3,activation='relu',padding='same')(pool2)
#     down3 = Conv2D(128,3,activation='relu',padding='same')(down3)
#     down3 = BatchNormalization()(down3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(down3)

#     bottleneck = Conv2D(264,3,activation='relu',padding='same')(pool3)
#     bottleneck = Conv2D(264,3,activation='relu',padding='same')(bottleneck)
#     bottleneck = BatchNormalization()(bottleneck)

#     unpool3 = UpSampling2D(size=(2,2))(bottleneck)
#     up3 = concatenate([unpool3,down3])
#     up3 = Conv2D(128,3,activation='relu',padding='same')(unpool3)
#     up3 = Conv2D(128,3,activation='relu',padding='same')(up3)
#     up3 = BatchNormalization()(up3)

#     unpool2 = UpSampling2D(size=(2,2))(up3)
#     up2 = concatenate([unpool2,down2])
#     up2 = Conv2D(64,3,activation='relu',padding='same')(unpool2)
#     up2 = Conv2D(64,3,activation='relu',padding='same')(up2)
#     up2 = BatchNormalization()(up2)

#     unpool1 = UpSampling2D(size=(2,2))(up2)
#     up1 = concatenate([unpool1,down1])
#     up1 = Conv2D(32,3,activation='relu',padding='same')(unpool1)
#     up1 = Conv2D(32,3,activation='relu',padding='same')(up1)
#     up1 = BatchNormalization()(up1)

#     correction = Conv2D(1,3,activation='relu',padding='same')(up1)

#     outputs = SubstracterLayer([ifft,correction])
    
#     model = Model(inputs=inputs,outputs=outputs)
#     return model

# def complex_but_not(input_shape=(200,200,2,1)):
    
#     inputs = Input(shape=input_shape)

#     Block1 = complexconv(inputs,filters = 128, kernel_size = (5,5))
#     Block1 = complexconv(Block1,filters=1, kernel_size = (5,5))
#     Block1 = BatchNormalization()(Block1)
#     Block1 = Add()([inputs,Block1])

#     Block2 = complexconv(Block1,filters = 128, kernel_size = (5,5))
#     Block2 = complexconv(Block2,filters=1, kernel_size = (5,5))
#     Block2 = BatchNormalization()(Block2)
#     Block2 = Add()([inputs,Block2])

#     ifft = IfftLayer(Block2)#ifft_function(inputs)#Lambda(function=ifft_function,output_shape=inputs.shape,input_shape=inputs.shape)(inputs)#Lambda(function=ifft_function,output_shape=Block2.shape,input_shape=Block2.shape)(Block2)

#     down1 = complexconv(ifft,filters = 32, kernel_size = 3)
#     down1 = complexconv(down1,filters = 32, kernel_size = 3)
#     down1 = BatchNormalization()(down1)
#     pool1 = complexmaxpool(down1)
    
#     down2 = complexconv(pool1,filters = 64, kernel_size = 3)
#     down2 = complexconv(down2,filters = 64, kernel_size = 3)
#     down2 = BatchNormalization()(down2)
#     pool2 = complexmaxpool(down2)

    
#     down3 = complexconv(pool2,filters = 128, kernel_size = 3)
#     down3 = complexconv(down3,filters = 128, kernel_size = 3)
#     down3 = BatchNormalization()(down3)
#     pool3 = complexmaxpool(down3)

#     down4 = complexconv(pool3,filters = 256, kernel_size = 3)
#     down4 = complexconv(down4,filters = 256, kernel_size = 3)
#     down4 = BatchNormalization()(down4)

#     unpool3 = complexupsample(down4)
#     up3 = concatenate([unpool3,down3])
#     up3 = complexconv(up3,filters = 128, kernel_size = 3)
#     up3 = complexconv(up3,filters = 128, kernel_size = 3)
#     up3 = BatchNormalization()(up3)

#     unpool2 = complexupsample(up3)
#     up2 = concatenate([unpool2,down2])
#     up2 = complexconv(up2,filters = 64, kernel_size = 3)
#     up2 = complexconv(up2,filters = 64, kernel_size = 3)
#     up2 = BatchNormalization()(up2)

#     unpool1 = complexupsample(up2)
#     up1 = concatenate([unpool1,down1])
#     up1 = complexconv(up1,filters = 32, kernel_size = 3)
#     up1 = complexconv(up1,filters = 32, kernel_size = 3)
#     up1 = BatchNormalization()(up1)

#     to_correct = complexconv(inputs,filters = 1, kernel_size = 3)#up1

#     outputs = SubstracterLayer([ifft,to_correct])


#     # outputs = AbsLayer(outputs)

#     model = Model(inputs=inputs,outputs=outputs)
#     return model

# def test_model(input_shape=(200,200,2,1)):
#     inputs = Input(shape=input_shape)
#     ifft = IfftLayer(inputs)
#     model = Model(inputs=inputs,outputs=ifft)
#     return model

def ifft_layer_old(kspace):
    real = Lambda(lambda kspace : kspace[:,:,:,0])(kspace)
    imag = Lambda(lambda kspace : kspace[:,:,:,1])(kspace)
    kspace_complex = tf.complex(real,imag)
    kspace_complex = tf.signal.ifftshift(kspace_complex,axes=[1,2])
    rec1 = tf.signal.ifft2d(kspace_complex)
    rec1 = tf.signal.fftshift(rec1,axes=[1,2])
    rec1 = [tf.math.real(rec1),tf.math.imag(rec1)]
    return tf.stack(rec1,axis=-1)# rec1 = tf.expand_dims(rec1, -1)
    # return tf.expand_dims(tf.abs(rec1),3)

def get_unet_old(input_shape=(200,200,2,1), n_filters=16, dropout=0.5, batchnorm=True, kernel_initializer="he_normal"):
    
#    print(input_img)
    # contracting path
#    input_img=(input_img-1000)/1000
    inputs = Input(shape=input_shape)

    c1 = conv2d_block_3(inputs, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block_3(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block_3(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block_3(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block_3(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block_3(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block_3(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block_3(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block_3(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
   
    outkspace = Conv2D(2, (1, 1), activation='linear') (c9)
    
    outkspace_comb = Add()([outkspace,inputs])

    
    img_rec = Lambda(ifft_layer_old)(outkspace_comb)
    
    d1 = conv2d_block_3(img_rec, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block_2(img_rec, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    k1 = MaxPooling2D((2, 2)) (d1)
    k1 = Dropout(dropout*0.5)(k1)

    d2 = conv2d_block_3(k1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block_2(k1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    k2 = MaxPooling2D((2, 2)) (d2)
    k2 = Dropout(dropout)(k2)

    d3 = conv2d_block_3(k2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block_2(k2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    k3 = MaxPooling2D((2, 2)) (d3)
    k3 = Dropout(dropout)(k3)

    d4 = conv2d_block_3(k3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block_2(k3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    k4 = MaxPooling2D(pool_size=(2, 2)) (d4)
    k4 = Dropout(dropout)(k4)
    
    d5 = conv2d_block_3(k4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block_2(k4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
    
    # expansive path
    v6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (d5)
    v6 = concatenate([v6, d4])
    v6 = Dropout(dropout)(v6)
    d6 = conv2d_block_3(v6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block_2(v6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)

    v7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (d6)
    v7 = concatenate([v7, d3])
    v7 = Dropout(dropout)(v7)
    d7 = conv2d_block_3(v7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block_2(v7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)

    v8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (d7)
    v8 = concatenate([v8, d2])
    v8 = Dropout(dropout)(v8)
    d8 = conv2d_block_3(v8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block_2(v8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)

    v9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (d8)
    v9 = concatenate([v9, d1], axis=3)
    v9 = Dropout(dropout)(v9)
    d9 = conv2d_block_3(v9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)#conv2d_block_2(v9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
   
    outimg = Conv2D(2, (1, 1), activation='linear') (d9)

    outimg_comb = Add()([img_rec,outimg])

    model = Model(inputs=[inputs], outputs=[outkspace_comb,outimg_comb])#
    model.compile(loss = [reduced_nrmse,reduced_nrmse],optimizer=Adam())#,tf.keras.losses.MeanAbsoluteError() 
    return model
