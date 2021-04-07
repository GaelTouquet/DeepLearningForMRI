import numpy as np
import tensorflow as tf

def absnorm_np(cplx_array):
    absval = np.abs(cplx_array)
    angle = np.angle(cplx_array)
    absval /= np.amax(absval)
    return absval * (np.cos(angle) + 1j*np.sin(angle))

def absnorm_tf(cplx_tf):
    the_abs = tf.math.abs(cplx_tf)
    the_angle = tf.math.angle(cplx_tf)
    factors = tf.math.reduce_max(the_abs,axis=[1,2])
    the_abs = tf.math.divide(the_abs,tf.expand_dims(tf.expand_dims(factors,axis=1),axis=2))
    real = the_abs * tf.math.cos(the_angle)
    imag = the_abs * tf.math.sin(the_angle)
    return tf.complex(real,imag)

def nothing(whatever):
    return whatever

def normalisation(name='absnorm',spe='nponly'):
    func_dict = {}
    if name=='absnorm':
        if spe=='nponly':
            func_dict['np'] = absnorm_np
            func_dict['tf'] = None
        else:
            func_dict['np'] = absnorm_np
            func_dict['tf'] = absnorm_tf
    elif not name:
        func_dict['np'] = None
        func_dict['tf'] = None
    else:
        raise ValueError('Name of normalisation unknown : {}'.format(name))
    return func_dict