import sys
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import average_precision_score

sys.path.append('../../Classes')
from DataHandler import DataHandler
from RFModelController import RFModelController

# selected model:

params =  {
    "n_estimators" : 175,
    "criterion" : "entropy",
    "max_depth" : 20,
    # "min_samples_split" : 0.0001,
    "min_samples_leaf" : 0.0005, 
    "max_features" : 0.4,
    "max_samples": 0.75,
    "verbose": 1
}

data = DataHandler(validation_sample= False, features_txt= 'all_features.txt', balance= 'weights')
data.main()

# cont = RFModelController(data =data, name = 'all_features')
# cont.main_model(model_params= params, permutation_train_max_samples= 300_000, permutation_test_max_samples= 300_000)

# i will use the same model for the feature sets that don't give good results
data.features_txt = 'all_features_abs_mags.txt'
data.features_labels()
cont = RFModelController(data =data, name = 'all_features_abs_mags')
cont.main_model(model_params= params, permutation_train_max_samples= 300_000, permutation_test_max_samples= 300_000)

data.features_txt = 'all_features_z_mass.txt'
data.features_labels()
cont = RFModelController(data =data, name = 'all_features_z_mass')
cont.main_model(model_params= params, permutation_train_max_samples= 300_000, permutation_test_max_samples= 300_000)

#---------------------------------------------------

# data = DataHandler(validation_sample= False, features_txt= 'all_features.txt', fields_list=['W06'])
# data.main()

# # for a first search
# search_param_distr =  {
#     "n_estimators" : [50, 100, 150, 200],
#     "criterion" : ["gini", "entropy"],
#     "max_depth" : [20, 50, 100, 300],
#     "min_samples_split" : [2, 0.05, 0.01, 0.001, 0.0001, 0.00005],
#     "min_samples_leaf" : [1, 0.05, 0.01, 0.001, 0.0001, 0.00005],
#     "max_features" : ["sqrt", "log2", 0.3],
#     "max_samples": [0.25, 0.5, 0.75]
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
#     "n_estimators" : [75, 100, 150, 175],
#     "criterion" : ["entropy", "gini"],
#     "max_depth" : [20, 50, 100],
#     "min_samples_split" : [0.01, 0.001, 0.0001, 0.00005],
#     "min_samples_leaf" : [1, 0.001, 0.0001, 0.00005],
#     "max_features" : ["sqrt", "log2", 0.3],
#     "max_samples": [0.25, 0.5, 0.75]
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
# name = "all_features/search-2"

# 3
# search_param_distr =  {
#     "n_estimators" : [100, 125, 150, 175],
#     "criterion" : ["entropy", "gini"],
#     "max_depth" : [20, 50, 100],
#     "min_samples_split" : [0.01, 0.001, 0.0001, 0.00005],
#     "min_samples_leaf" : [0.001, 0.005, 0.0001, 0.00005],
#     "max_features" : [0.2, 0.3, 0.4],
#     "max_samples": [0.5, 0.75]
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
# name = "all_features/search-3"

# #  4... 
# search_param_distr =  {
#     "n_estimators" : [100, 125, 150, 175],
#     "criterion" : ["entropy", "gini"],
#     "max_depth" : [20, 30, 50],
#     "min_samples_split" : [0.01, 0.001, 0.0001],
#     "min_samples_leaf" : [0.001, 0.002, 0.0005],
#     "max_features" : [0.4],
#     "max_samples": [0.5, 0.75]
# }
# search_params = dict(
#     cv = 2, 
#     n_jobs = -1,  
#     verbose = 10,
#     n_iter = 70, 
#     scoring = ['average_precision'],
#     refit =  'average_precision',
#     error_score = 'raise',
# )
# search_class = "RandomizedSearchCV"
# name = "all_features/search-4"


# a last search...
# search_param_distr =  {
#     "n_estimators" : [125, 150, 175],
#     "criterion" : ["entropy", "gini"],
#     "max_depth" : [20, 30, 50],
#     "min_samples_split" : [0.001, 0.0001],
#     "min_samples_leaf" : [0.001, 0.0005],
#     "max_features" : [0.4],
#     "max_samples": [0.5, 0.75]
# }
# search_params = dict(
#     cv = 4, 
#     n_jobs = -1,  
#     verbose = 10,
#     scoring = ['average_precision', 'neg_log_loss'],
#     refit =  'average_precision',
#     error_score = 'raise',
# )
# search_class = "GridSearchCV"
# name = "all_features/search-5"

# cont = RFModelController(data = data, name = name, model = RandomForestClassifier(bootstrap= True, n_jobs = 1, verbose= 0, class_weight= 'balanced_subsample', oob_score= average_precision_score))
# cont.main_search(search_param_distr= search_param_distr, search_params= search_params, search_class= search_class, plot_search= False)