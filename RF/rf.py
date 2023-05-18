"""
Random forests hyper parameter search and model testing.

From command line: python rf.py [-m model.json] [-s search.json] [-r 42]
                                [--min_n500 0] [--max_n500 60] [--min_z 0] [--max_z 1]

Options:
    -m     Train and test a model with hyper parameters given in  a JSON file. Save metrics.
    -s     Search on hyper parameters, with distributions given in  a JSON file. Save metrics for best model.
    -r     Change random_state for training-testing split (it's set to a fixed number by default)
    --min_n500, --max_n500    Minimum and maximum for cluster's n500. Default: None
    --min_z, --max            Minimum and maximum for cluster's z. Default: None

Test results are saved into a "metrics" directory. Searches results are saved into "search_results".
The features to be used are listed in features.txt
JSON files have to be in the same directory as this script.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_data():
    # read and join tables
    df1 = pd.read_csv('../DATA/clean-HSC-unWISE-W01.csv')
    df1 = df1.drop(columns = [f for f in df1.columns if ('isnull' in f)])
    df2 = pd.read_csv('../DATA/clean-HSC-unWISE-W02.csv')
    df2 = df2.drop(columns = [f for f in df2.columns if ('isnull' in f)])
    df = pd.concat([df1,df2], axis = 'rows')
    
    # add colors
    df['gr'] = df['g_cmodel_mag'] - df['r_cmodel_mag']
    df['ri'] = df['r_cmodel_mag'] - df['i_cmodel_mag']
    df['iz'] = df['i_cmodel_mag'] - df['z_cmodel_mag']
    df['zy'] = df['z_cmodel_mag'] - df['y_cmodel_mag']
    
    return df
       
def features_labels(df):  
    
    # read features list
    with open('../DATA/features1.txt') as file:
        feat = file.read().splitlines()
    lab = 'member'
    
    # print number of members
    n_mem = df[df.member == 1].shape[0]
    n_no = df[df.member == 0].shape[0]
    n = df.shape[0]
    print ('Members: {} ({:.2f}%)'.format(n_mem, n_mem/n*100))
    print ('Non members: {} ({:.2f}%)'.format(n_no, n_no/n*100))
    print('-'*70)
    
    return feat,lab

def z_n500_limits(df, mini_z = None, maxi_z = None, mini_n500 = None, maxi_n500 = None):
    # Select galaxies near clusters in a z and n500 range...
    # df has a column 'id_cl_near', with the id of the nearest cluster. 
    df_cl = pd.read_table('../DATA/clusters.dat', delim_whitespace=True, usecols=[0,3,4,5,9,11,12], names=['id_cl','ra_cl','dec_cl','phot_z_cl', 'r500_cl','mass_cl','n500_cl'])
    
    cond = True
    if mini_z != None:
        cond = (cond) & (df_cl.phot_z_cl >= mini_z)
    if maxi_z != None:
        cond = (cond) & (df_cl.phot_z_cl <= maxi_z)
    if mini_n500 != None:  
        cond = (cond) & (df_cl.n500_cl >= mini_n500)
    if maxi_n500 != None:
        cond = (cond) & (df_cl.n500_cl <= maxi_n500)
        
    df_cl = df_cl[cond]  
    df = df[df['id_cl_near'].isin(df_cl.id_cl)]
    
    return df
    
def split(df, lab, ran_state):
    
    # split into training and testing samples. 
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size = 0.3, stratify = df[lab], random_state = ran_state)
    
    print ('Training: {} members, {} non members'.format(train[train.member == 1].shape[0], train[train.member == 0].shape[0]))
    print ('Testing: {} members, {} non members'.format(test[test.member == 1].shape[0], test[test.member == 0].shape[0]))
    print('-'*70)
    
    return train,test

def undersample(df, feat, lab):
    # Undersample majority class (gridsearch takes so long....)
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy= 4./6., replacement= False)
    rus_feat, rus_lab = rus.fit_resample(df[feat], df[lab])
    rus_feat[lab] = rus_lab

    n_mem = rus_feat[rus_feat.member == 1].shape[0]
    n_no = rus_feat[rus_feat.member == 0].shape[0]
    n = rus_feat.shape[0]
    print ('Members after undersampling...: {} ({:.2f}%)'.format(n_mem, n_mem/n*100))
    print ('Non members after undersampling...: {} ({:.2f}%)'.format(n_no, n_no/n*100))
    del n,n_mem,n_no

    return rus_feat

def write_report(model, test, feat, lab, filename):
    
    # write metrics to file: score, auc, classification report
    from sklearn.metrics import auc, classification_report, roc_curve, precision_recall_curve
    
    pred = model.predict(test[feat])  
    scores = model.predict_proba(test[feat])
    model_score = model.score(test[feat], test[lab])

    fpr, tpr, thres_roc = roc_curve(test[lab], scores[:,1], pos_label=1)
    prec, rec, thres_pr = precision_recall_curve(test[lab], scores[:,1], pos_label= 1)

    with open(filename, mode='w') as file:
        
        for key in model.get_params():
            file.write(f'{key}: {model.get_params()[key]} \n')
        file.write('-'*70 + '\n')
        file.write('Model score: {:.4g} \n'.format(model_score))
        file.write('ROC curve AUC: {}\n'.format(auc(fpr, tpr)))
        file.write('Precision-recall AUC: {}\n'.format(auc(rec, prec)))
        file.write('-'*70 + '\n')
        file.write(classification_report(test[lab],pred))
    
    print ('Model score: {:.4g} \n'.format(model_score))
    print('ROC curve AUC: {}\n'.format(auc(fpr, tpr)))
    print('Precision-recall AUC: {}\n'.format(auc(rec, prec)))
    print('-'*70)
    print(classification_report(test[lab],pred))

def plot_report(model, test, feat, lab, filename):
    
    # plot roc curve, precision-recall curve and confusion matrix to file
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
    
    pred = model.predict(test[feat])  
    scores = model.predict_proba(test[feat])

    fpr, tpr, thres_roc = roc_curve(test[lab], scores[:,1], pos_label=1)
    prec, rec, thres_pr = precision_recall_curve(test[lab], scores[:,1], pos_label= 1)
    
    plt.figure(figsize=(16,4))
        
    # confusion matrix
    plt.subplot(1, 3, 1)
    conf_m = confusion_matrix(test[lab], pred)
    df_conf_m = pd.DataFrame(conf_m, index=[0,1], columns=[0,1])
    sns.heatmap(df_conf_m, cmap=sns.color_palette('light:teal', as_cmap=True), annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # ROC curve
    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid()
    
    # Precision-recall curve
    a = len (test[test[lab]==1])/len(test)
    plt.subplot(1, 3, 3)
    plt.plot(rec, prec)
    plt.plot([0, 1], [a, a] , color='gray', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.xlim([0,1])
    plt.ylim([0,1])

    plt.savefig(filename, dpi=150, bbox_inches= 'tight') 

def plot_importances(model, feat, filename):
    importances = np.sort(model.feature_importances_)[::-1]
    sorted_feat = [x for _,x in sorted(zip(model.feature_importances_, feat))][::-1]

    plt.figure(figsize= (17, 10))
    plt.grid()
    sns.barplot(x = importances, y = sorted_feat)
    plt.savefig(filename, dpi=150, bbox_inches= 'tight') 

def params_search(json_file, model, df, feat, lab):
    
    # read parameters from json file
    import json
    with open(json_file, mode= 'r') as f:
        params = f.read()
        json_dict = json.loads(params)
        params = json_dict['hyperparams']
        search_class = json_dict['search_class']
        search_params = json_dict['search_params']

    # search instantiation and fitting
    # halving searches try every model with a reduced number of samples, select the best models, and repeat with more samples
    # i don't think they work well for this problem....
    match search_class:
        case 'HalvingGridSearchCV':
            from sklearn.experimental import enable_halving_search_cv
            from sklearn.model_selection import HalvingGridSearchCV
            search_model = HalvingGridSearchCV(estimator= model, param_grid= params)
        case 'HalvingRandomSearchCV':
            from sklearn.experimental import enable_halving_search_cv
            from sklearn.model_selection import HalvingRandomSearchCV
            search_model = HalvingRandomSearchCV(estimator= model, param_distributions= params)
        case 'GridSearchCV':
            from sklearn.model_selection import GridSearchCV
            search_model = GridSearchCV(estimator= model, param_grid= params) 
        case 'RandomizedSearchCV':
            from sklearn.model_selection import RandomizedSearchCV
            search_model = RandomizedSearchCV(estimator= model, param_distributions= params) 
        case _:
            raise ValueError("Invalid value for 'search_class'") 
    search_model.set_params(**search_params)
    search_model.fit(df[feat], df[lab])

    # save and print results
    df_res = pd.DataFrame(search_model.cv_results_).sort_values('rank_test_score')
    df_res.to_csv(f'search_results/{search_json[:-5]}.csv')
    print('-'*70)
    print(f"Best model: score {search_model.best_score_}")
    print(search_model.best_estimator_)
    # print('-'*70)
    # if 'Halving' in search_class:
    #     print(df_res[df_res.iter == search_model.n_iterations_].sort_values('rank_test_score'))
    # else:
    #     print(df_res.sort_values('rank_test_score').head(15)) 

    return search_model.best_estimator_

def train_test_model(json_file, model, df, feat, lab):
    # read hyper parameters from json file
    import json
    with open(json_file, mode= 'r') as f:
        params = f.read()
        params = json.loads(params)
    
    # set parameters and fit
    model.set_params(**params)
    model.fit(df[feat], df[lab])

    return model
            


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', action='store', default=None) # test a model? Stores name of JSON file with params.
    parser.add_argument('-s', '--search', action='store', default=None,) # search on hyperparameters? Stores name of JSON file with params.
    parser.add_argument('-r', '--random_state', action='store', default=42, type= int) # for train-test split
    parser.add_argument('--min_n500', action='store', default=None, type= float) # minimum for cluster n500?
    parser.add_argument('--max_n500', action='store', default=None, type= float) # maximum for cluster n500?
    parser.add_argument('--min_z', action='store', default=None, type= float) # minimum for cluster z?
    parser.add_argument('--max_z', action='store', default=None, type= float) # maximum for cluster z?
    
    model_json = parser.parse_args().model
    search_json = parser.parse_args().search   
    random_state = parser.parse_args().random_state
    min_n500 = parser.parse_args().min_n500
    max_n500 = parser.parse_args().max_n500
    min_z = parser.parse_args().min_z
    max_z = parser.parse_args().max_z
    
    # prepare data
    data = read_data()
    if any(val != None for val in [min_n500, max_n500, min_z, max_z]):
        data = z_n500_limits(data, min_z, max_z, min_n500, max_n500) 
    features,label = features_labels(df=data)
    data = undersample(data, features, label) # this is the only balancing strategy i have tried so far...
    training,testing = split(df=data, lab=label, ran_state=random_state)
    
    # make model
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(bootstrap= True, n_jobs = -1, verbose= 0, class_weight= 'balanced') 
    
    # search on hyper parameters?
    if search_json:
        rf_model = params_search(search_json, rf_model, data, features, label)
        write_report(rf_model, testing, features, label, f'metrics/{search_json[:-5]}.txt')
        plot_report(rf_model, testing, features, label, f'metrics/{search_json[:-5]}.png')
        plot_importances(rf_model, features, f'metrics/importances_{search_json[:-5]}.png')

    # test a model?
    if model_json:
        rf_model = train_test_model(model_json, rf_model, data, features, label)
        write_report(rf_model, testing, features, label, f'metrics/{model_json[:-5]}.txt')
        plot_report(rf_model, testing, features, label, f'metrics/{model_json[:-5]}.png')
        plot_importances(rf_model, features, f'metrics/importances_{model_json[:-5]}.png')