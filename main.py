# from MRI.DataTypes.KSpace.KData import KData

from NN.Agent import Agent
from NN.Inputs import preprocess_data, DataGenerator, RandomMask, test_data
from NN.architectures import myunet as unet
from NN.architectures import convolutional_autoencoder
import tensorflow as tf
# from tensorflow.image import ssim
import numpy as np
import os
import numpy as np
import time
np.random.seed(123)  # for reproducibility


def ssim(y1,y2):
    return tf.image.ssim(y1,y2,max_val=1)

# def my_ssim(y1,y2):
#     return ssim(y1,y2)

batch_size = 5
epochs = 20
acceleration = 10
coil_type = 'single' # 'single' or 'multi'
fraction = 1
tag = 'propertest'
input_mask = RandomMask(acceleration=acceleration, seed=0xdeadbeef)
model = unet()#tf.keras.models.load_model('D:\\NN_DATA\\singlecoil_acc1_unittest2\\agentNov_05_16_50',custom_objects={'ssim':ssim})#unet()

timestamp = time.strftime("%h_%d_%H_%M")

name = '{}coil_acc{}_{}'.format(coil_type,acceleration,tag)
print('\nusing Directory :\nD:\\NN_DATA\\{}\n'.format(name))

print('preparing train data')
preprocess_data('D:\\fastMRI_DATA\\{}coil_train\\'.format(coil_type),
                name, 'train', input_mask=input_mask,multicoil=(coil_type=='multi'),normalise=True,fraction=fraction)
print('preparing val data')
preprocess_data('D:\\fastMRI_DATA\\{}coil_val\\'.format(coil_type),
                name, 'val', input_mask=input_mask,multicoil=(coil_type=='multi'),normalise=True,fraction=fraction)
print('preparing test data')
preprocess_data('D:\\fastMRI_DATA\\{}coil_test_v2\\'.format(coil_type),
                name, 'test{}'.format(timestamp), input_mask=input_mask,multicoil=(coil_type=='multi'),normalise=True,fraction=fraction)


print('preparing generators')
train_gen = DataGenerator(name.format(
    'train'), cat='train', batch_size=batch_size)
val_gen = DataGenerator(name.format('val'), cat='val', batch_size=batch_size)
# test_gen = DataGenerator(name.format('val'),batch_size=batch_size)


print('compiling model')
# myunet = unet(input_size=(320, 320, 1))


model.compile(loss='binary_crossentropy',
               optimizer='adadelta',
               metrics=['mse',ssim])

myagent = Agent(model, train_gen, val_gen, name)
print('starting training')
myagent.train(epochs=epochs, verbose=1)
print('starting testing')
test_data('D:\\NN_DATA\\{}\\test{}'.format(name,timestamp),myagent)
print('saving agent')
myagent.save('D:\\NN_DATA\\{}\\agent{}'.format(name,timestamp))