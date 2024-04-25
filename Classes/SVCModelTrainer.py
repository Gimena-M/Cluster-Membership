from DataHandler import DataHandler
from ModelTrainer import ModelTrainer

from cuml.svm import SVC

class SVCModelTrainer(ModelTrainer):

    def __init__(self, data: DataHandler, name: str,
                 model = SVC(class_weight= 'balanced', cache_size = 4000, verbose = False)):
        self.model = model
        self.data = data
        self.name = name

    def _select_search_class(self, search_param_distr: dict, search_class: str):
        match search_class:
            case 'GridSearchCV':
                from cuml.model_selection import GridSearchCV
                search_model = GridSearchCV(estimator = self.model, param_grid= search_param_distr)
            case 'RandomizedSearchCV':
                raise NotImplementedError('RandomizedSearchCV not implemented')
            case _:
                raise ValueError("Invalid value for 'search_class'")
            
        return search_model