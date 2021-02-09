import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Conv2DTranspose, Dropout, Input, Conv2D, Lambda,MaxPooling2D, concatenate, UpSampling2D,Add, BatchNormalization, Dense
from tensorflow.keras.optimizers import Adam

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

def norm_abs(image):
    k_abs = image[:,:,:,0]
    k_angle = image[:,:,:,1]
    return tf.stack([1e5*k_abs,k_angle],axis=-1)

def unnorm_abs(image):
    k_abs = image[:,:,:,0]
    k_angle = image[:,:,:,1]
    return tf.stack([k_abs/1e5,k_angle],axis=-1)


def ifft_layer(kspace):
    k_abs = kspace[:,:,:,0]
    k_angle = kspace[:,:,:,1]
    real = k_abs * tf.math.cos(k_angle)
    imag = k_abs * tf.math.sin(k_angle)
    kspace_complex = tf.complex(real,imag)
    kspace_complex = tf.signal.ifftshift(kspace_complex,axes=[1,2])
    rec1 = tf.signal.ifft2d(kspace_complex)
    rec1 = tf.signal.fftshift(rec1,axes=[1,2])
    rec1 = [tf.math.abs(rec1),tf.math.angle(rec1)]
    return tf.stack(rec1,axis=-1)

def dense_block(input_tensor,n_layers,n_filters):
    x = input_tensor
    import pdb;pdb.set_trace()
    for i in range(n_layers):
        x = Dense(n_filters)(x)
    Dense(256*256*2)(x)
    return x

def simple_dense(n_layers,n_filters,input_shape=(256,256,2),fullskip=True):
    inputs = Input(shape=input_shape)
    x = dense_block(inputs,n_layers,n_filters)
    tf.reshape(x,(256,256,2))
    if fullskip:
        x = Add()([x,inputs])

    x = Lambda(ifft_layer)(x)

    model = Model(inputs=inputs,outputs=x)
    return model


def conv2d_block(input_tensor, n_filters, kernel_size=3, n_layer=1, batchnorm=True,kernel_initializer="he_normal"):
    x = input_tensor
    for i in range(n_layer):
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer=kernel_initializer,
            padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x

def testmodel(input_shape=(256,256,2)):
    inputs = Input(shape=input_shape)
    x = Conv2D(filters=2, kernel_size=(3,3),padding="same")(inputs)
    model = Model(inputs=inputs,outputs=x)
    return model

def add_unet(input_layer, depth=5, n_layer=3,n_filters=16, kernel_size=3,
dropout=0.5, batchnorm=True, kernel_initializer="he_normal", normfunction=lambda x: x, 
unormfunc=None,fullskip=False):

    normed_inputs = Lambda(normfunction)(input_layer)

    downstream = []

    x = normed_inputs

    for i in range(depth):
        x = conv2d_block(x, n_layer=n_layer,n_filters=n_filters*(2**i),
        kernel_size=kernel_size, batchnorm=batchnorm, kernel_initializer=kernel_initializer)
        downstream.append(x)
        x = MaxPooling2D((2, 2)) (x)
        x = Dropout(dropout)(x)
    
    x = conv2d_block(x, n_layer=n_layer,n_filters=n_filters*(2**(depth+1)),
    kernel_size=kernel_size, batchnorm=batchnorm, kernel_initializer=kernel_initializer)

    for i in range(depth):
        x = Conv2DTranspose(n_filters*(2**(depth-(i+1))),
        kernel_size, strides=(2, 2), padding='same') (x)
        x = concatenate([x, downstream[depth-(i+1)]])
        x = Dropout(dropout)(x)
        x = conv2d_block(x, n_layer=n_layer, n_filters=n_filters*(2**(depth-(i+1))), 
        kernel_size=kernel_size, batchnorm=batchnorm, kernel_initializer=kernel_initializer)

    x = Conv2D(input_layer.shape[-1], (1, 1), activation='linear') (x)

    if fullskip:
        x = Add()([x,normed_inputs])

    if unormfunc:
        x = Lambda(unormfunc)(x)

    return x

def get_unet(input_shape=(256,256,2), n_layer=3, n_filters=16, normfunction=lambda x: x,
dropout=0.5, batchnorm=True, kernel_initializer="he_normal",fullskip=False,kernel=3,
unormfunc=None, depth=5):

    inputs = Input(shape=input_shape)

    unet = add_unet(inputs,n_layer=n_layer,
    dropout=dropout,batchnorm=batchnorm,kernel_initializer=kernel_initializer,
    normfunction=normfunction,fullskip=fullskip,unormfunc=unormfunc,depth=depth,n_filters=n_filters,kernel_size=kernel)

    model = Model(inputs=inputs, outputs=unet)
    return model

def get_unet_fft(input_shape=(256,256,2), n_layer=3, n_filters=16, normfunction=lambda x: x,
dropout=0.5, batchnorm=True, kernel_initializer="he_normal",fullskip=False,
unormfunc=None, depth=5,kernel=3):

    inputs = Input(shape=input_shape)

    unet = add_unet(inputs,n_layer=n_layer,
    dropout=dropout,batchnorm=batchnorm,kernel_initializer=kernel_initializer,
    normfunction=normfunction,fullskip=fullskip,unormfunc=unormfunc,depth=depth,n_filters=n_filters,kernel_size=kernel)

    img = Lambda(ifft_layer)(unet)

    model = Model(inputs=inputs, outputs=img)
    return model

def get_wnet(input_shape=(256,256,2), n_layer_kspace=3, n_layer_img=3, n_filters=16, 
dropout=0.5, batchnorm=True, kernel_initializer="he_normal", 
normfunction_kspace=lambda x: x, normfunction_img=lambda x: x, fullskip=True):

    inputs = Input(shape=input_shape)

    kspace_unet = add_unet(inputs,n_layer=n_layer_kspace,
    dropout=dropout,batchnorm=batchnorm,kernel_initializer=kernel_initializer,
    normfunction=normfunction_kspace,fullskip=fullskip)

    img = Lambda(ifft_layer)(kspace_unet)

    img_unet = add_unet(img,n_layer=n_layer_img,
    dropout=dropout,batchnorm=batchnorm,kernel_initializer=kernel_initializer,
    normfunction=normfunction_img,fullskip=fullskip)

    model = Model(inputs=[inputs], outputs=[kspace_unet,img_unet])
    return model