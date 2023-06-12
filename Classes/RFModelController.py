"""
This class handles data preparing, model training and model testing.
It uses the DataHandler, RFModelTrainer and RFModelTester classes.

Initialization arguments are:
    * data (DataHandler): An instance of DataHandler. It does not need to have already read or prepared the data.
    * name (str): A name for the created files.
    * model (RandomForestClassifier): A base RF model. (default: RandomForestClassifier(bootstrap= True, n_jobs = -1, verbose= 0, class_weight= 'balanced'))

The search_main() method performs a hyperparameter search. It has the following arguments:
    * search_param_distr (dict): Distribution of hyperparameters for a hyperparameter search.
    * search_params (dict): Parameters for the search object.
    * search_class (str): Name of the search class to be used. Can be 'GridSearchCV', 'RandomizedSearchCV', 'HalvingGridSearchCV' or 'HalvingRandomSearchCV'
    * read_data (bool): Read and prepare the data (run data.main())? (default: False)
    * prep_data (bool): If the data has already been read, prepare the data (run data.prep())? (default: False)
    * thresholds (list or None): List of thresholds to be tested. (default: None)
    * best_thresholds (bool): Compute thresholds that maximize F-Score or G-Means? (default: False)
By default it assumes that data has already been read and prepared.

The model_main() method can train and test a model. It has the following arguments:
    * model_params (dict): Parameters for the model. (default: {})
    * read_data (bool): Read and prepare the data (run data.main())? (default: False)
    * prep_data (bool): If the data has already been read, prepare the data (run data.prep())? (default: False)
    * thresholds (list or None): List of thresholds to be tested. (default: None)
    * best_thresholds (bool): Compute thresholds that maximize F-Score or G-Means? (default: False)
    * sort_importances (bool): Sort features in descending order of importance when plotting feature importances? (default: True)
By default it assumes that data has already been read and prepared, and that the model has not been trained.

"""

import sys

from sklearn.ensemble import RandomForestClassifier
sys.path.append('../../Classes')
from DataHandler import DataHandler
from RFModelTester import RFModelTester
from RFModelTrainer import RFModelTrainer

class RFModelController:

    def __init__(self, data: DataHandler, name: str, 
                 model: RandomForestClassifier = RandomForestClassifier(bootstrap= True, n_jobs = -1, verbose= 0, class_weight= 'balanced')):
        self.name = name
        self.data = data
        self.model = model

    def main_search(self, search_param_distr: dict, search_params: dict, search_class: str,
               read_data: bool = False, prep_data: bool = False,
               thresholds: list|None = None, best_thresholds: bool = False):
        
        if read_data:
            self.data.main()
        elif prep_data:
            self.data.prep()

        self.trainer = RFModelTrainer(data = self.data, name = self.name, model = self.model)
        self.trainer.params_search(search_param_distr= search_param_distr, search_params= search_params, search_class= search_class, name = self.name)

        self.tester = RFModelTester(model= self.trainer.model, data= self.data, name= self.name)
        self.tester.main(extra_args= self.trainer.args(), thresholds= thresholds, best_thresholds=best_thresholds)
    
    def main_model(self, model_params: dict = {}, model_exists: bool = False,
              read_data: bool = False, prep_data: bool = False,
              thresholds: list|None = None, best_thresholds: bool = False, sort_importances: bool = True):

        if read_data:
            self.data.main()
        elif prep_data:
            self.data.prep()

        self.trainer = RFModelTrainer(data = self.data, name = self.name, model = self.model)
        
        if model_exists:
            self.trainer.load_model()
        else:
            self.trainer.train_model(model_params=model_params)

        self.tester = RFModelTester(model= self.trainer.model, data= self.data, name= self.name)
        self.tester.main(extra_args= self.trainer.args(), thresholds= thresholds, best_thresholds=best_thresholds, sort_importances= sort_importances)
