"""
Superclass to perform model training (inherits to RFModelTrainer and NNModelTrainer).
It is not meant to be instantiated.

Some attributes are:
    * name (str): Name to be used in file names with metrics and saved models.
    * model: NN, RF or SVC model.
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

import pandas as pd

class ModelTrainer:

    # name: str
    # model 
    # data: DataHandler

    def params_search(self, search_param_distr: dict, search_params: dict, search_class: str, name:str = "search"):

        # search instantiation and fitting
        # halving searches try every model with a reduced number of samples, select the best models, and repeat with more samples
        # i don't think they work well for this problem....
        match search_class:
            case 'HalvingGridSearchCV':
                from sklearn.model_selection import HalvingGridSearchCV
                search_model = HalvingGridSearchCV(estimator= self.model, param_grid= search_param_distr)
            case 'HalvingRandomSearchCV':
                from sklearn.model_selection import HalvingRandomSearchCV
                search_model = HalvingRandomSearchCV(estimator= self.model, param_distributions= search_param_distr)
            case 'GridSearchCV':
                from sklearn.model_selection import GridSearchCV
                search_model = GridSearchCV(estimator= self.model, param_grid= search_param_distr)
            case 'RandomizedSearchCV':
                from sklearn.model_selection import RandomizedSearchCV
                search_model = RandomizedSearchCV(estimator= self.model, param_distributions= search_param_distr)
            case _:
                raise ValueError("Invalid value for 'search_class'")
        search_model.set_params(**search_params)
        search_model.fit(self.data.training_features(), self.data.training_labels())

        # save and print results
        df_res = pd.DataFrame(search_model.cv_results_).sort_values('rank_test_score')
        df_res.to_csv(f'search_results/{name}.csv', index= False)
        print('-'*70)
        print(f"Best model: score {search_model.best_score_}")
        print(search_model.best_estimator_)
        # print('-'*70)
        # if 'Halving' in search_class:
        #     print(df_res[df_res.iter == search_model.n_iterations_].sort_values('rank_test_score'))
        # else:
        #     print(df_res.sort_values('rank_test_score').head(15)) 

        self.model =  search_model.best_estimator_
        self.save_model()
        return self.model

    def train_model(self, model_params: dict):
        self.model.set_params(**model_params)
        self.model.fit(self.data.training_features(), self.data.training_labels())
        self.save_model()
        return self.model
    
    def load_model(self):
        import joblib
        self.model = joblib.load(f'saved_models/{self.name}.joblib')
    
    def save_model(self):
        import joblib
        joblib.dump(self.model, f'saved_models/{self.name}.joblib')

    def args(self):
        return self.model.get_params()

