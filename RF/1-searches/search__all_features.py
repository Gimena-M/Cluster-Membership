import sys
# from sklearn.ensemble import RandomForestClassifier

sys.path.append('../../Classes')
from DataHandler import DataHandler
from RFModelController import RFModelController

# selected model:

params =  {
    "n_estimators" : 250,
    "criterion" : "entropy",
    "max_depth" : 50,
    "min_samples_split" : 20,
    "min_samples_leaf" : 15, 
    "max_features" : "sqrt",
    "verbose": 10
}


data = DataHandler(validation_sample= False, features_txt= 'all_features.txt', balance= 'weights')
data.main()

cont = RFModelController(data =data, name = 'all_features')
cont.main_model(model_params= params, permutation_train_max_samples= 120_000)

# i will use the same model for the feature sets that don't give good results
data.features_txt = 'all_features_abs_mags.txt'
data.features_labels()
cont = RFModelController(data =data, name = 'all_features_abs_mags')
cont.main_model(model_params= params, permutation_train_max_samples= 120_000)

data.features_txt = 'all_features_z_mass.txt'
data.features_labels()
cont = RFModelController(data =data, name = 'all_features_z_mass')
cont.main_model(model_params= params, permutation_train_max_samples= 120_000)

#---------------------------------------------------

# data = DataHandler(validation_sample= False, features_txt= 'all_features.txt', fields_list=['W06'])
# data.main()

# # for a first search
# search_param_distr =  {
#     "n_estimators" : [20, 50, 100, 150],
#     "criterion" : ["gini", "entropy"],
#     "max_depth" : [None, 10, 20, 50],
#     "min_samples_split" : [2, 5, 10, 20],
#     "min_samples_leaf" : [1, 5, 10, 100],
#     "max_features" : ["sqrt", "log2"]
# }
# search_params = dict(
#     cv = 2, 
#     n_jobs = -1,  
#     verbose = 4,
#     n_iter = 100, 
#     scoring = ['average_precision', 'neg_log_loss'],
#     refit =  'average_precision',
#     error_score = 'raise',
# )
# search_class = "RandomizedSearchCV"
# name = "all_features/search-1"



# # a second search...
# search_param_distr =  {
#     "n_estimators" : [150, 175, 200],
#     "criterion" : ["entropy", "gini"],
#     "max_depth" : [None, 20, 50, 80],
#     "min_samples_split" : [5, 10],
#     "min_samples_leaf" : [1,10,20,50], 
#     "max_features" : ["sqrt"]
# }
# search_params = dict(
#     cv = 2, 
#     n_jobs = -1,  
#     verbose = 4,
#     n_iter = 50, 
#     scoring = ['average_precision', 'neg_log_loss'],
#     refit =  'average_precision',
#     error_score = 'raise',
# )
# search_class = "RandomizedSearchCV"
# name = "all_features/search-2"

# # a last search...
# search_param_distr =  {
#     "n_estimators" : [200, 225, 250],
#     "criterion" : ["entropy"],
#     "max_depth" : [None, 20, 50, 80],
#     "min_samples_split" : [5, 10, 20],
#     "min_samples_leaf" : [15, 20, 25], 
#     "max_features" : ["sqrt"]
# }
# search_params = dict(
#     cv = 4, 
#     n_jobs = -1,  
#     verbose = 4,
#     scoring = ['average_precision', 'neg_log_loss'],
#     refit =  'average_precision',
#     error_score = 'raise',
# )
# search_class = "GridSearchCV"
# name = "all_features/search-3"

# cont = RFModelController(data = data, name = name, model = RandomForestClassifier(bootstrap= True, n_jobs = 1, verbose= 0, class_weight= 'balanced_subsample'))
# cont.main_search(search_param_distr= search_param_distr, search_params= search_params, search_class= search_class)