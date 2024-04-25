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
                 model = RandomForestClassifier(bootstrap= True, n_jobs = -1, verbose= 0, class_weight= 'balanced_subsample')):
        self.model = model
        self.data = data
        self.name = name

    def _select_search_class(self, search_param_distr: dict, search_class: str):
         # halving searches try every model with a reduced number of samples, select the best models, and repeat with more samples
        # i don't think they work well for this problem....
        match search_class:
            # case 'HalvingGridSearchCV':
            #     from sklearn.model_selection import HalvingGridSearchCV
            #     search_model = HalvingGridSearchCV(estimator= self.model, param_grid= search_param_distr)
            # case 'HalvingRandomSearchCV':
            #     from sklearn.model_selection import HalvingRandomSearchCV
            #     search_model = HalvingRandomSearchCV(estimator= self.model, param_distributions= search_param_distr)
            case 'GridSearchCV':
                from sklearn.model_selection import GridSearchCV
                search_model = GridSearchCV(estimator= self.model, param_grid= search_param_distr)
            case 'RandomizedSearchCV':
                from sklearn.model_selection import RandomizedSearchCV
                search_model = RandomizedSearchCV(estimator= self.model, param_distributions= search_param_distr)
            case _:
                raise ValueError("Invalid value for 'search_class'")
            
        return search_model
    
    def train_model(self, model_params: dict = ..., warm_start: bool = False):
        model_params['warm_start'] = warm_start
        return super().train_model(model_params, warm_start)