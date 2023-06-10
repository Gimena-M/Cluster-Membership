"""
Class to perform neural network models training (inherits from ModelTrainer).

Arguments for instantiation are:
    * name (str): Name to be used in file names with metrics and saved models.
    * data (DataHandler): instance of DataHandler.
    * layers (list): List of layers for the model (do not include a normalization layer). (default = [])
    * compile_params (dict): Dictionary with parameters for model.compile(). Includes optimizer, loss function and metrics. (default = {})
    * epochs (int): Number of epochs for training. (default: 60)
    * normalization (bool): Include a normalization layer? (default: True)
    * weights (bool): Include class weights from DataHandler? (default: True)
    * verbose (int): Verbosity lever for model.fit. (default: 2)

The params_search method does not work here... :3
The train_model method does not take any arguments. The model and trainig history are saved to a 'saved_models' directory.

Model and training history can be loaded from 'saved_models' with the load_model method.

"""


from DataHandler import DataHandler
from ModelTrainer import ModelTrainer

import pandas as pd
import tensorflow as tf

class NNModelTrainer(ModelTrainer):

    def __init__(self, data: DataHandler, name: str, 
                 layers: list = [], compile_params: dict = {}, epochs: int = 60, 
                 normalization: bool = True, weights: bool = True, verbose: int = 2):
        self.name = name
        self.layers = layers
        self.compile_params = compile_params
        self.epochs = epochs
        self.normalization = normalization
        self.data = data
        self.weights = weights
        self.verbose = verbose

    def args(self):
        # return dictionary with some attributes.
        d = dict(
            epochs = self.epochs,
            normalization = self.normalization,
            weights = self.weights
        )
        return d

    def params_search(self):
        raise NotImplementedError("Hyperparameter search not implemented")

    def train_model(self):

        callbacks = [
            # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            tf.keras.callbacks.CSVLogger(filename = f'saved_models/{self.name}_log.csv'),
            tf.keras.callbacks.ModelCheckpoint(filepath = f'saved_models/{self.name}.h5', monitor = 'val_loss',  save_best_only = True),
            # tf.keras.callbacks.TensorBoard()
            ]

        self.normalize()

        # make model and compile. Make dict with arguments for fit function.
        self.model = tf.keras.Sequential(self.layers)
        self.model.compile(**self.compile_params)
        fit_params = dict(
            x = self.data.training_features().values,
            y = self.data.training_labels().values,
            verbose = self.verbose,
            callbacks = callbacks,
            validation_data = (self.data.validation_features().values, self.data.validation_labels().values),
            epochs = self.epochs,
            batch_size = 4096
            )

        # class imbalance?
        if self.weights:
            fit_params['class_weight'] = self.data.weights

        self.model.fit(**fit_params)
        self.load_model()

    def load_model(self):
        # load best model and history from files
        self.model = tf.keras.models.load_model(f'saved_models/{self.name}.h5')
        self.history = pd.read_csv(f'saved_models/{self.name}_log.csv')

    def save_model(self):
        pass

    def normalize(self):
        #include normalization layer?
        if self.normalization:
            norm = tf.keras.layers.Normalization(input_shape=(len(self.data.features),))
            norm.adapt(data = self.data.training_features().values)
            self.layers.insert(0, norm)