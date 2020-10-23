import os
import numpy as np
import h5py

class NNSet_base(object):
    """
    Class that holds information on all the sets used at every level to train/test a network.
    """

    def inputs(self):
        return np.reshape(self._inputs,(self._inputs.shape[0],)+self.input_shape)

    def outputs(self):
        return np.reshape(self._outputs,(self._outputs.shape[0],)+self.input_shape)

    def predictions(self):
        return np.reshape(self._predictions,(self._prediction[0],)+self.input_shape)

    def update_predictions(self, predictions):
        self._predictions = predictions

    def save(self, path):
        h5file = h5py.File(path,'w')
        h5file.create_dataset('inputs',data=self.inputs())
        h5file.create_dataset('outputs',data=self.outputs())
        if hasattr(self, '_predictions'):
            h5file.create_dataset('predictions',self.predictions)

class NNSet_from_fastMRI(NNSet_base):
    """
    Class to make a NNSet from the original h5 files from fastMRI.
    """

    def __init__(self, pairs_of_input_outputs, n_channels=1):
        """
        pairs_of_inputs = set of [input,output]
        """
        self._inputs = np.zeros((len(pairs_of_input_outputs),n_channels)+pairs_of_input_outputs[0][0].shape,dtype = pairs_of_input_outputs[0][0].dtype)
        self._outputs = np.zeros((len(pairs_of_input_outputs),n_channels)+pairs_of_input_outputs[0][0].shape,dtype = pairs_of_input_outputs[0][0].dtype)

class NNSet(NNSet_base):
    """
    docstring
    """
    
    def __init__(self,path,input_shape):
        """
        docstring
        """
        self.input_shape = input_shape
        if '.h5' in path:
            f = h5py.File(path,'r')
            self._inputs = f['inputs']
            self._outputs = f['outputs']
            if 'predictions' in f:
                self._predictions = f['predictions']
        else:
            files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(f,path) and '.h5' in f)]
            forsize = h5py.File(os.path.join(files[0],path),'r')['inputs']
            self._inputs = np.zeros((len(files))+forsize.shape, dtype = forsize.dtype)
            self._outputs = np.zeros((len(files))+forsize.shape, dtype = forsize.dtype)
            self._predictions = np.zeros((len(files))+forsize.shape, dtype = forsize.dtype)
            for i, f in enumerate(files):
                tmp_f = h5py.File(os.path.join(f,path),'r')
                self._inputs[i] = tmp_f['inputs']
                self._outputs[i] = tmp_f['outputs']
                if 'predictions' in tmp_f:
                    self._predictions[i] = tmp_f['predictions']
