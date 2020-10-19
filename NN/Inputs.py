import numpy as np

class NNSet(object):
    """
    Class that holds information on all the sets used at every level to train/test a network.
    """
    pass

    def __init__(self, pairs_of_input_outputs):
        """
        pairs_of_inputs = set of [input,output]
        """
        self.sets = []
        self._inputs = []
        self._outputs = []
        self._predictions = []
        for pair in pairs_of_input_outputs:
            self.sets.append([pair[0],pair[1],None])
            self._inputs.append(pair[0])
            self._outputs.append(pair[1])

    def inputs(self):
        return np.array(self._inputs)

    def outputs(self):
        return np.array(self._outputs)

    def predictions(self):
        return np.array(self._predictions)

    def update_predictions(self, prediction_list):
        for i in range(len(prediction_list)):
            self._predictions[i] = prediction_list[i]
            self.sets[i][2] = prediction_list[i]