from tensorflow.keras import backend as keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, concatenate, UpSampling2D, Lambda
from tensorflow.keras.models import Model

def convolutional_autoencoder(pretrained_weights=None, input_shape=(320,320,1)):
    input = Input(shape=input_shape)

    # encoding #
    x = Conv2D(filters = 128, kernel_size = (5,5), activation='relu', padding = 'same')(input)
    x = MaxPooling2D(pool_size = (2,2), padding = 'same')(x)

    x = Conv2D(filters = 64, kernel_size = (5,5), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size = (2,2), padding = 'same')(x)

    x = Conv2D(filters = 64, kernel_size = (5,5), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size = (2,2), padding = 'same')(x)

    # Decoding #

    x = Conv2D(64, (5,5), activation = 'relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)

    x = Conv2D(64, (5,5), activation = 'relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)

    x = Conv2D(128, (5,5), activation = 'relu')(x)
    x = UpSampling2D((2,2))(x)
    output = Conv2D(1, (5,5), activation='sigmoid', padding='same')(x)
    model = Model(inputs=input,outputs=output)
    return model


def myunet(pretrained_weights=None, input_size=(320, 320, 1)):
    # inputs = Input(input_size)
    # conv1 = Conv2D(64,3, activation='relu')(inputs)
    # conv1 = Conv2D(64,3, activation='relu')(conv1)
    # pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    # conv2 = Conv2D(128,3, activation='relu')(pool1)
    # conv2 = Conv2D(128,3, activation='relu')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    # conv3 = Conv2D(128,3, activation='relu')(pool2)
    # conv3 = Conv2D(128,3, activation='relu')(conv3)
    # pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    # outputs = Conv2D(320,3, activation='sigmoid')(pool3)
    # model = Model(inputs=inputs,outputs=outputs)
    # if(pretrained_weights):
    #     model.load_weights(pretrained_weights)
    # return model

    #This version's best score on single coils: ssim=0.9625 val_ssim=0.944 epochs=30 
    inputs = Input(input_size)
    conv1 = Conv2D(64, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    # drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 3, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 3, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 3, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 3, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 5, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    outputs = Conv2D(1, 1, activation='sigmoid', padding='same',
            kernel_initializer='he_normal')(conv9)
    # outputs = Lambda(lambda x: x*256)(outputs)
    model = Model(inputs=inputs,outputs=outputs)
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

# def unet(pretrained_weights=None, input_size=(256, 256, 1)):
#     inputs = Input(input_size)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(inputs)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(pool1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(pool2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(pool3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

#     conv5 = Conv2D(1024, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(pool4)
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)

#     up6 = Conv2D(512, 2, activation='relu', padding='same',
#                  kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
#     merge6 = concatenate([drop4, up6], axis=3)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(merge6)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(conv6)

#     up7 = Conv2D(256, 2, activation='relu', padding='same',
#                  kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
#     merge7 = concatenate([conv3, up7], axis=3)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(merge7)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(conv7)

#     up8 = Conv2D(128, 2, activation='relu', padding='same',
#                  kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
#     merge8 = concatenate([conv2, up8], axis=3)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(merge8)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(conv8)

#     up9 = Conv2D(64, 2, activation='relu', padding='same',
#                  kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
#     merge9 = concatenate([conv1, up9], axis=3)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(merge9)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(conv9)
#     conv9 = Conv2D(2, 3, activation='relu', padding='same',
#                    kernel_initializer='he_normal')(conv9)
#     conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

#     model = Model(inputs=inputs, outputs=conv10)

#     model.compile(optimizer=Adam(lr=1e-4),
#                   loss='binary_crossentropy', metrics=['accuracy'])

#     # model.summary()

#     if(pretrained_weights):
#         model.load_weights(pretrained_weights)

#     return model