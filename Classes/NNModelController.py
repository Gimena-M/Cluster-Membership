"""
This class handles data preparing, model training and model testing.
It uses the DataHandler, NNModelTrainer and NNModelTester classes.

Initialization arguments are:
    * data (DataHandler): An instance of DataHandler. It does not need to have already read or prepared the data.
    * name (str): A name for the created files.
    * layers (list): List of layers for the model (do not include a normalization layer). (default = [])
    * compile_params (dict): Dictionary with parameters for model.compile(). Includes optimizer, loss function and metrics. (default = {})
    * epochs (int): Number of epochs for training. (default: 60)
    * normalization (bool): Include a normalization layer? (default: True)
    * weights (bool): Include class weights from DataHandler? (default: True)
    * verbose (int): Verbosity lever for model.fit. (default: 2)

The main() method has the following optional arguments:
    * model_exists (bool): Has the model already been trained and saved? (default: False)
    * read_data (bool): Read and prepare the data (run data.main())? (default: False)
    * prep_data (bool): If the data has already been read, prepare the data (run data.prep())? (default: False)
    * optimize_threshold (bool): Use decision threshold that maximizes F1-score? (default: True)
    * loss_lims (tuple): y limits for a loss vs epochs plot. (default: (None, None))
By default it assumes that data has already been read and prepared, and that the model has not been trained.

"""

import sys
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelTester import NNModelTester
from NNModelTrainer import NNModelTrainer

class NNModelController:

    def __init__(self, data: DataHandler, name: str, 
                 layers: list = [], compile_params: dict = {}, epochs: int = 1000, normalization: bool = True, weights: bool = True, verbose: int = 2, patience: int = 20, batch_size: int = 4096):
        self.data = data
        self.layers = layers
        self.name = name
        self.compile_params = compile_params
        self.epochs = epochs
        self.normalization = normalization
        self.weights = weights
        self.verbose = verbose
        self.patience = patience
        self.batch_size = batch_size

    def main(self, model_exists: bool = False, resume_training: bool = False, 
             read_data: bool = False, prep_data: bool = False,
             loss_lims: tuple = (None,None), optimize_threshold: bool = True,
             test_name: str|None = None, retrain_name: str|None = None,
             importances: list|None = ['permutation_train', 'permutation_test'], sort_importances: str|None = 'permutation_train', 
             permutation_train_max_samples: int|float = 200_000, permutation_test_max_samples: int|float = 200_000
             ):
        #thresholds: list|None = None, best_thresholds: bool = False, 
        
        if read_data:
            self.data.main()
        elif prep_data:
            self.data.prep()

        self.trainer = NNModelTrainer(data = self.data, name = self.name, 
                                      layers = self.layers, compile_params= self.compile_params,
                                      epochs = self.epochs, normalization= self.normalization, weights= self.weights, verbose= self.verbose, patience= self.patience, batch_size= self.batch_size)
        
        if model_exists:
            self.trainer.load_model()
            if resume_training:
                if retrain_name: 
                    self.name = retrain_name
                    self.trainer.name = retrain_name
                self.trainer.train_model()
        else:
            self.trainer.make_model()
            self.trainer.train_model()

        test_name = test_name if test_name else self.name

        self.tester = NNModelTester(model= self.trainer.model, data= self.data, name= test_name, history= self.trainer.history)
        self.tester.main(optimize_threshold= optimize_threshold, extra_args= self.trainer.args(), loss_lims=loss_lims, importances= importances, sort_importances= sort_importances, permutation_test_max_samples= permutation_test_max_samples, permutation_train_max_samples= permutation_train_max_samples) #, thresholds= thresholds, best_thresholds=best_thresholds, 