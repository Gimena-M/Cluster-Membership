This directory contains some tests:

+ 1-searches: performed random searches for a model with all features, with and without the nearest BCG's redshift as a feature.
+ 2-z_steps_0.1: trained a model with BCGs in redshift ranges of size 0.1. 
+ 2-z_steps_0.05: trained a model with BCGs in redshift ranges of size 0.05.

Each directory contains:

+ A python script
+ A "metrics" folder with results.
+ A "saved_models" folder with models.
+ A "search_results" folder with search results, if a search was performed.

A basic script to test a model would be:

    # Imports from Classes
    import sys
    sys.path.append('../../Classes')
    from DataHandler import DataHandler
    from RFModelController import RFModelController

    model_params =  {
        "n_estimators" : 120,
        "criterion" : "entropy",
        "max_depth" : 10,
        "min_samples_split" : 2, 
        "min_samples_leaf" : 2,
        "max_features" : "log2"
    }
    name = "model"

    # Read and prepare data, train and test model
    data = DataHandler(validation_sample= False, features_txt= 'all_features.txt', fields_list=['W01', 'W02', 'W03', 'W04'])
    cont = RFModelController(data = data.copy(), name = name)
    cont.main_model(model_params= model_params, read_data= True)

A basic script to search on hyperparameters and test the best model would be:
    
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
    name = "random-search__bcg"

    data = DataHandler(validation_sample= False, features_txt= 'all_features_bcg.txt', fields_list=['W01', 'W02', 'W03', 'W04'])
    cont = RFModelController(data = data, name = name)
    cont.main_search(search_param_distr= search_param_distr, search_params= search_params, search_class= search_class, read_data= True)
