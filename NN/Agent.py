import h5py
import tensorflow.keras as keras
import tensorflow as tf

class Agent(object):
    """
    Class to hold informations about an Agent.
    """
    def __init__(self, architecture, train_generator, val_generator, name):
        """
        architecture = keras model
        train_generator = generator that will handle the training phase supply of data
        val_generator = generator that supplies the validation data
        name = name of the directory
        """
        self.network = architecture
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.logdir = 'D:\\NN_DATA\\{}\\log'.format(name)

    def train(self, batch_size, epochs, verbose=0):
        """
        Train the model.
        """
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir, histogram_freq=1)
        self.history = self.network.fit_generator(self.train_generator,validation_data=self.val_generator,epochs=5,callbacks=[tensorboard_callback])

    def process(self, test_generator,verbose=0):
        """
        Predict the output of the model evaluated on the data supplied by test_generator.
        """
        return self.network.predict_generator(test_generator, verbose=verbose)

    def evaluate(self, evaluation_set):
        """
        Evaluate the network on a given set
        """
        self.evaluation_set = evaluation_set
        self.evaluation = self.network.evaluate(evaluation_set.inputs(),evaluation_set.outputs())

    def save(self, path):
        """
        Saves the network at the given path.
        """
        self.network.save(path)

    def load(self, path):
        """
        Loads the network saved at the given path.
        """
        self.network = keras.models.load_model(path)