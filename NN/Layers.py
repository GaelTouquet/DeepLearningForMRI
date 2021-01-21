import tensorflow as tf
from tensorflow.keras.layers import Layer

def abs_func(x):
    return tf.math.abs(tf.dtypes.complex(x[0],x[1]))


def ifft_function(x):
    complexes = tf.signal.ifft2d(tf.dtypes.complex(x[0],x[1]))
    return tf.stack([tf.math.real(complexes),tf.math.imag(complexes)],axis=-2)


class ComplexConvAdder(Layer):

    def call(self, realfromreal, realfromcmplx, cmplxfromreal,cmplxfromcmplx):
        newreal = tf.math.subtract(realfromreal,realfromcmplx)
        newcplx = tf.math.add(cmplxfromreal,cmplxfromcmplx)
        return newreal, newcplx


class ComplexConv2D(Layer):
    def __init__(self,filters,kernel_size,activation):
        super(ComplexConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        if any([x%2!=1 for x in self.kernel_size]):
            raise ValueError('Even kernel shapes not implemented yet! Sorry!')
        self.activation = activation

    def build(self, input_shape):
        if input_shape[-2]!=2:
            raise ValueError("input_shape[-2] is the complex dimension, it should always be 2.")
        self.w = self.add_weight(
            shape=self.kernel_size+(self.filters,self.input_shape[-1],2),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.filters,2),
            initializer="random_normal",
            trainable=True
        )
    
    def call(self, inputs):
        real = tf.Tensor()