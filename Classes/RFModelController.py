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
    * test (bool): Test the best model? (default: False)
    * optimize_threshold (bool): Use decision threshold that maximizes F1-score? (default: True)
    * importances (list|None): List of importances to compute. If None, don't compute importances. (default: ['gini', 'permutation_test', 'gini'])
    * sort_importances (str|None): Sort features in descending order of gini or permutation importance when plotting feature importances? Takes values None (no sorting), 'or importance name (default: 'gini')
    * permutation_train_max_samples (int|float): max_samples for permutation_importance with training sample (default: 1.0), 
    * permutation_test_max_samples (int|float): max_samples for permutation_importance with test sample (default: 1.0), 
By default it assumes that data has already been read and prepared.

The model_main() method can train and test a model. It has the following arguments:
    * model_params (dict): Parameters for the model. (default: {})
    * read_data (bool): Read and prepare the data (run data.main())? (default: False)
    * prep_data (bool): If the data has already been read, prepare the data (run data.prep())? (default: False)
    * model_exists (bool): Load existing model? (default: False), 
    * resume_training (bool): Use warm start to train model, if another model exists? (default: False)
    * retrain_name (str|None): Name of retrained model. If None, use old model's name. (default: None),
    * test (bool): Test model? (default: True)
    * test_name (str|None): Name for metrics files. If None, use model's name. (default: None)
    * optimize_threshold (bool): Use decision threshold that maximizes F1-score? (default: True)
    * importances (list|None): List of importances to compute. If None, don't compute importances. (default: ['permutation_train', 'permutation_test', 'gini'])
    * sort_importances (str|None): Sort features in descending order of gini or permutation importance when plotting feature importances? Takes values None (no sorting), or importance name (default: 'gini')
    * permutation_train_max_samples (int|float): max_samples for permutation_importance with training sample (default: 1.0), 
    * permutation_test_max_samples (int|float): max_samples for permutation_importance with test sample (default: 1.0), 
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
                 model: RandomForestClassifier = RandomForestClassifier(bootstrap= True, n_jobs = -1, verbose= 0, class_weight= 'balanced_subsample')):
        self.name = name
        self.data = data
        self.model = model

    def main_search(self, search_param_distr: dict, search_params: dict, search_class: str,
               read_data: bool = False, prep_data: bool = False,
               test: bool = False, optimize_threshold: bool = True, 
               importances: list|None = ['permutation_train', 'permutation_test', 'gini'], sort_importances: str|None = 'gini', 
                permutation_train_max_samples: int|float = 1.0, permutation_test_max_samples: int|float = 1.0):
        
        if read_data:
            self.data.main()
        elif prep_data:
            self.data.prep()

        self.trainer = RFModelTrainer(data = self.data, name = self.name, model = self.model)
        self.trainer.params_search(search_param_distr= search_param_distr, search_params= search_params, search_class= search_class, name = self.name)
        
        if test:
            self.tester = RFModelTester(model= self.trainer.model, data= self.data, name= self.name)
            self.tester.main(extra_args= self.trainer.args(), optimize_threshold = optimize_threshold,
                             importances= importances, sort_importances= sort_importances, permutation_train_max_samples= permutation_train_max_samples, permutation_test_max_samples= permutation_test_max_samples)
    
    def main_model(self, model_params: dict = {}, model_exists: bool = False, 
              read_data: bool = False, prep_data: bool = False,
              optimize_threshold: bool = True, test: bool = True, test_name: str|None = None,
              resume_training: bool = False, retrain_name: str|None = None,
              importances: list|None = ['permutation_train', 'permutation_test', 'gini'], sort_importances: str|None = 'gini', permutation_train_max_samples: int|float = 1.0, permutation_test_max_samples: int|float = 1.0):

        if read_data:
            self.data.main()
        elif prep_data:
            self.data.prep()

        self.trainer = RFModelTrainer(data = self.data, name = self.name, model = self.model)

        if model_exists:
            self.trainer.load_model()
            if resume_training:
                if retrain_name: 
                    self.name = retrain_name
                    self.trainer.name = retrain_name
                self.trainer.train_model(warm_start = True)
        else:
            self.trainer.train_model(model_params=model_params)

        if test:
            test_name = test_name if test_name else self.name
            self.tester = RFModelTester(model= self.trainer.model, data= self.data, name= test_name)
            self.tester.main(extra_args= self.trainer.args(), optimize_threshold = optimize_threshold, importances= importances, sort_importances= sort_importances, permutation_train_max_samples= permutation_train_max_samples, permutation_test_max_samples= permutation_test_max_samples)
