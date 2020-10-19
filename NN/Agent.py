class Agent(object):
    """
    Class to hold informations about an Agent.
    """
    def __init__(self, architecture, NNset):
        """
        docstring
        """
        self.network = architecture
        self.NNset = NNset
        self.NNset.agent = self

    def train(self, batch_size, epochs, verbose=0):
        """
        docstring
        """
        self.history = self.network.fit(self.NNset.inputs(),self.NNset.outputs(),batch_size=batch_size,epochs=epochs,verbose=verbose)

    def process(self, test_set,verbose=0):
        """
        docstring
        """
        self.NNset.update_predictions(self.network.predict(test_set, verbose=verbose))

    def evaluate(self, evaluation_set):
        """
        docstring
        """
        self.evaluation_set = evaluation_set
        self.evaluation = self.network.evaluate(evaluation_set.inputs(),evaluation_set.outputs())