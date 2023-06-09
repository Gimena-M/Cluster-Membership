import sys
sys.path.append('../../Classes')
from DataHandler import DataHandler
from RFModelController import RFModelController

search_param_distr =  {
    "n_estimators" : [20,50,80,100,120],
    "criterion" : ["gini", "entropy", "log_loss"],
    "max_depth" : [None, 10, 20, 30],
    "min_samples_split" : [2, 5, 10, 15],
    "min_samples_leaf" : [1, 2, 4, 10],
    "max_features" : ["sqrt", "log2"]
}
search_params = {
    "cv": 3, 
    "n_jobs": -1,  
    "verbose": 4,
    "n_iter": 100
    }
search_class = "RandomizedSearchCV"


# Search with all_features.txt
name = "random-search"
data = DataHandler(validation_sample= False, features_txt= 'all_features.txt', fields_list=['W03'])

cont = RFModelController(data = data, name = name)
cont.main_search(search_param_distr= search_param_distr, search_params= search_params, search_class= search_class, 
                 read_data= True)



# Search with all_features_bcg.txt
name = "random-search---bcg"
data.features_txt= 'all_features_bcg.txt'

cont = RFModelController(data = data, name = name)
cont.main_search(search_param_distr= search_param_distr, search_params= search_params, search_class= search_class, 
                 prep_data= True)




