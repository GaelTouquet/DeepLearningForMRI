import os
import h5py
import tensorflow.keras as keras
import tensorflow as tf


class Agent(object):
    """
    Class to hold informations about an Agent.
    """

    def __init__(self, architecture, train_generator, val_generator, workdir, timestamp):
        """
        architecture = keras model
        train_generator = generator that will handle the training phase supply of data
        val_generator = generator that supplies the validation data
        name = name of the directory
        """
        self.workdir = workdir
        self.network = architecture
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.logdir = os.path.join(self.workdir,'log')
        self.timestamp = timestamp

    def train(self, epochs, verbose=0):
        """
        Train the model.
        """
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.logdir, histogram_freq=1)
        if not os.path.isdir(os.path.join(self.workdir,'trainingsaves_{}'.format(self.timestamp))):
            os.mkdir(os.path.join(self.workdir,'trainingsaves_{}'.format(self.timestamp)))
        save_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.workdir,'trainingsaves_{}'.format(self.timestamp),'epoch{epoch:02d}.h5'), 
            period=1, save_best_only=False,save_weights_only=True
        )
        self.history = self.network.fit_generator(
            self.train_generator, validation_data=self.val_generator, epochs=epochs, callbacks=[tensorboard_callback,save_callback])#

    def test(self, test_set, batch_size=5, verbose=0):
        """
        Predict the output of the model evaluated on the data supplied by test_generator.
        """
        return self.network.predict(test_set, batch_size=batch_size, verbose=verbose)

    def evaluate(self, evaluation_set):
        """
        Evaluate the network on a given set
        """
        self.evaluation_set = evaluation_set
        self.evaluation = self.network.evaluate(
            evaluation_set.inputs(), evaluation_set.outputs())

    def save(self, path):
        """
        Saves the network at the given path.
        """
        self.network.save(path)

    def load_weights(self, path):
        """
        Loads the newtork weights at the given path.
        """
        self.network.load_weights(path)

    def load(self, path):
        """
        Loads the network saved at the given path.
        """
        self.network = keras.models.load_model(path)
