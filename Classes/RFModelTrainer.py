"""
Class to perform random forest models' training (inherits from ModelTrainer).

Arguments for instantiation are:
    * name (str): Name to be used in file names with metrics and saved models.
    * model: RF model.
    * data: instance of DataHandler.

The params_search method performs a search on hyperparameters for RF and SVC. Its arguments are:
    * search_param_distr (dict): Hyperparameter distributions.
    * search_params (dict): parameters for the search object.
    * search_class (str): can be 'GridSearchCV', 'RandomizedSearchCV', 'HalvingGridSearchCV' or 'HalvingRandomSearchCV'
    * name (str): name for the .csv file where results are saved, and for the files with metrics for the best model.
Search results are saved in a 'search_results' directory. Best model is saved to a 'saved_models' directory.

The train_model method trains a RF or SVC model. The only argument is:
    * model_params (dict): Parameters for the model.
The model is saved to a 'saved_models' directory.

A model can be loaded from 'saved_models' with the load_model method.

"""

from DataHandler import DataHandler
from ModelTrainer import ModelTrainer

from sklearn.ensemble import RandomForestClassifier

class RFModelTrainer(ModelTrainer):

    def __init__(self, data: DataHandler, name: str,
                 model = RandomForestClassifier(bootstrap= True, n_jobs = -1, verbose= 0, class_weight= 'balanced')):
        self.model = model
        self.data = data
        self.name = name