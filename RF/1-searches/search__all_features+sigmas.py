import sys
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import average_precision_score

sys.path.append('../../Classes')
from DataHandler import DataHandler
from RFModelController import RFModelController

# selected model:

params =  {
    "n_estimators" : 400,
    "criterion" : "entropy",
    "max_depth" : 20,
    # "min_samples_split" : 0.0001,
    "min_samples_leaf" : 0.0001, 
    "max_features" : 0.4,
    "max_samples": 0.75,
    "verbose": 1
}

data = DataHandler(validation_sample= False, features_txt= 'all_features_sigmas.txt', balance= 'weights')
data.main()

cont = RFModelController(data =data, name = 'all_features_sigmas')
cont.main_model(model_params= params, permutation_train_max_samples= 300_000, permutation_test_max_samples= 300_000)

#---------------------------------------------------

# data = DataHandler(validation_sample= False, features_txt= 'all_features_sigmas.txt', fields_list=['W06'], balance= 'weights')
# data.main()

# first search
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
#     verbose = 10,
#     n_iter = 100, 
#     scoring = ['average_precision', 'neg_log_loss'],
#     refit =  'average_precision',
#     error_score = 'raise',
# )
# search_class = "RandomizedSearchCV"
# name = "all_features_sigmas/search-1"

# # second search
# search_param_distr =  {
#     "n_estimators" : [150, 200, 250, 300],
#     "criterion" : ["gini", "entropy"],
#     "max_depth" : [20, 50, 80, 100],
#     "min_samples_split" : [0.001, 0.0005, 0.0001, 0.00005],
#     "min_samples_leaf" : [0.005, 0.0001, 0.00005],
#     "max_features" : ["sqrt", 0.2, 0.3, 0.4],
#     "max_samples": [0.25, 0.5, 0.75]
# }
# search_params = dict(
#     cv = 2, 
#     n_jobs = -1,  
#     verbose = 10,
#     n_iter = 100, 
#     scoring = ['average_precision', 'roc_auc'],
#     refit =  'average_precision',
#     error_score = 'raise',
# )
# search_class = "RandomizedSearchCV"
# name = "all_features_sigmas/search-2"

# 3rd search
# search_param_distr =  {
#     "n_estimators" : [250, 300, 350],
#     "criterion" : ["entropy"],
#     "max_depth" : [20, 50, 80],
#     "min_samples_split" : [0.0005, 0.0001, 0.00005],
#     "min_samples_leaf" : [0.0001, 0.00005],
#     "max_features" : [0.3, 0.4],
#     "max_samples": [0.5, 0.75]
# }
# search_params = dict(
#     cv = 2, 
#     n_jobs = -1,  
#     verbose = 10,
#     n_iter = 50, 
#     scoring = ['average_precision', 'roc_auc'],
#     refit =  'average_precision',
#     error_score = 'raise',
# )
# search_class = "RandomizedSearchCV"
# name = "all_features_sigmas/search-3"


# 4
# search_param_distr =  {
#     "n_estimators" : [400],
#     "criterion" : ["entropy"],
#     "max_depth" : [20, 50, 80],
#     "min_samples_split" : [0.0005, 0.0001],
#     "min_samples_leaf" : [0.0001],
#     "max_features" : [0.3, 0.4],
#     "max_samples": [0.75]
# }
# search_params = dict(
#     cv = 4, 
#     n_jobs = -1,  
#     verbose = 10,
#     scoring = ['average_precision', 'roc_auc'],
#     refit =  'average_precision',
#     error_score = 'raise',
# )
# search_class = "GridSearchCV"
# name = "all_features_sigmas/search-4"

# cont = RFModelController(data = data, name = name, model = RandomForestClassifier(bootstrap= True, n_jobs = 1, verbose= 0, class_weight= 'balanced_subsample', oob_score= average_precision_score))
# cont.main_search(search_param_distr= search_param_distr, search_params= search_params, search_class= search_class, plot_search= False)
