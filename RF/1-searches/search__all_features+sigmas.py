import sys
# from sklearn.ensemble import RandomForestClassifier

sys.path.append('../../Classes')
from DataHandler import DataHandler
from RFModelController import RFModelController

# selected model:

params =  {
    "n_estimators" : 300,
    "criterion" : "entropy",
    "max_depth" : 100,
    "min_samples_split" : 5,
    "min_samples_leaf" : 5, 
    "max_features" : "sqrt",
    "verbose": 10,
    
}


data = DataHandler(validation_sample= False, features_txt= 'all_features_sigmas.txt', balance= 'weights')
data.main()

cont = RFModelController(data =data, name = 'all_features_sigmas')
cont.main_model(model_params= params, sort_importances= 'permutation', optimize_threshold= True)

#---------------------------------------------------

# data = DataHandler(validation_sample= False, features_txt= 'all_features_sigmas.txt', fields_list=['W06'], balance= 'weights')
# data.main()

# # first search
# search_param_distr =  {
#     "n_estimators" : [30, 80, 150, 200, 300],
#     "criterion" : ["gini", "entropy"],
#     "max_depth" : [None, 10, 50, 80],
#     "min_samples_split" : [2, 5, 10, 20],
#     "min_samples_leaf" : [1, 10, 70, 100],
#     "max_features" : ["sqrt", "log2"]
# }
# search_params = dict(
#     cv = 2, 
#     n_jobs = -1,  
#     verbose = 4,
#     n_iter = 70, 
#     scoring = ['average_precision', 'neg_log_loss'],
#     refit =  'average_precision',
#     error_score = 'raise',
# )
# search_class = "RandomizedSearchCV"
# name = "all_features_sigmas/search-1"

# # second search
# search_param_distr =  {
#     "n_estimators" : [120, 180, 250, 300],
#     "criterion" : ["entropy"],
#     "max_depth" : [None, 50, 100, 200],
#     "min_samples_split" : [5, 10, 20, 30],
#     "min_samples_leaf" : [1, 5, 10, 15],
#     "max_features" : ["sqrt", "log2"]
# }
# search_params = dict(
#     cv = 2, 
#     n_jobs = -1,  
#     verbose = 4,
#     n_iter = 60, 
#     scoring = ['average_precision', 'neg_log_loss'],
#     refit =  'average_precision',
#     error_score = 'raise',
# )
# search_class = "RandomizedSearchCV"
# name = "all_features_sigmas/search-2"

# # third search
# search_param_distr =  {
#     "n_estimators" : [250, 300, 350],
#     "criterion" : ["entropy"],
#     "max_depth" : [None, 200, 300],
#     "min_samples_split" : [5, 10, 20],
#     "min_samples_leaf" : [1, 5, 10],
#     "max_features" : ["sqrt"]
# }
# search_params = dict(
#     cv = 3, 
#     n_jobs = -1,  
#     verbose = 10,
#     scoring = ['average_precision', 'neg_log_loss'],
#     refit =  'average_precision',
#     error_score = 'raise',
# )
# search_class = "GridSearchCV"
# name = "all_features_sigmas/search-3"


# cont = RFModelController(data = data, name = name, model = RandomForestClassifier(bootstrap= True, n_jobs = 1, verbose= 0, class_weight= 'balanced_subsample'))
# cont.main_search(search_param_distr= search_param_distr, search_params= search_params, search_class= search_class)
