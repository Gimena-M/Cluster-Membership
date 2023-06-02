"""
Create, train and test a NN model that (attempts to) preddict cluster membership.

From command line: python model_tests.py model.py [-e] [-f W01 W02 W03] [-r 42] [-t] [-bt]
                                        [--min_n500 0] [--max_n500 60] [--min_z 0] [--max_z 1]
                                        [--feat_max feature value] [--feat_min feature value]


Arguments:
    .py file with model parameters (has to be in the same directory as this file).
Options:
    -e     If model has already been saved and trained, load existing model and training history.
    -f     List of HSC fields (default: W01, W02, W03 & W04)
    -r     Change random_state for training-validation-testing split (it's set to a fixed number by default)
    -t     Test different thresholds
    -bt    Compute thresholds that maximize F-Score or G-Means
    --min_n500, --max_n500    Minimum and maximum for cluster's n500. Default: None
    --min_z, --max            Minimum and maximum for cluster's z. Default: None
    --feat_max                Limit feature to max. value? (Can be used more than once)
    --feat_min                Limit feature to min. value? (Can be used more than once)

The .py file with model parameters has:
    layers: list of layers for the network
    compile_params: parameters for model.compile(). Optimizer, loss function, metrics...
    epochs: number of epochs
    normalization: if True, a normalization layer is added as a first layer, and adapted with features.
    balance: how to deal with class imbalance. Can be 'weights', 'SMOTE' or None

Results are saved into a "metrics" directory.
The features to be used are listed in features.txt
"""



import pandas as pd
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(fields):
    # read and join tables
    li = []
    for fi in fields:
        d = pd.read_csv(f'../DATA/clean-HSC-unWISE-{fi}.csv')
        li.append(d.drop(columns = [f for f in d.columns if ('isnull' in f)]))
    df = pd.concat(li, axis = 'rows')
    
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
    # Select clusters in a z and n500 range...
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
    # df = df[df['id_cl_near'].isin(df_cl.id_cl)]
    df['id_cl_near'] = [eval(s) for s in df['id_cl_near']]
    df = df[[any([id in df_cl['id_cl'].values for id in gal.id_cl_near]) for _,gal in df.iterrows()]]
    
    return df

def feature_limits(df: pd.DataFrame, max_dict: dict, min_dict: dict):
    conds = []
    for key in max_dict.keys():
        conds.append((df[key] <= float(max_dict[key])))
    for key in min_dict.keys():
        conds.append((df[key] >= float(min_dict[key])))
    
    for c in conds:
        df = df[c]

    # print number of members
    n_mem = df[df.member == 1].shape[0]
    n_no = df[df.member == 0].shape[0]
    n = df.shape[0]
    print ('Members after feature limits: {} ({:.2f}%)'.format(n_mem, n_mem/n*100))
    print ('Non members feature limits: {} ({:.2f}%)'.format(n_no, n_no/n*100))
    print('-'*70)
    return df

def split(df, lab, ran_state):
    
    # split into training, testing and validation samples. 
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size = 0.3, stratify = df[lab], random_state = ran_state)
    val, test = train_test_split(test, test_size = 0.3, stratify = test[lab], random_state = ran_state)
    
    print ('Training: {} members, {} non members'.format(train[train.member == 1].shape[0], train[train.member == 0].shape[0]))
    print ('Validation: {} members, {} non members'.format(val[val.member == 1].shape[0], val[val.member == 0].shape[0]))
    print ('Testing: {} members, {} non members'.format(test[test.member == 1].shape[0], test[test.member == 0].shape[0]))
    print('-'*70)
    
    return train,val,test
    
def write_report(pred, model: tf.keras.Sequential, test, feat, lab, filename, args):
    
    # write metrics into file: loss, auc, classification report
    from sklearn.metrics import auc, classification_report, roc_curve, precision_recall_curve
    
    pred_classes = np.round(pred, decimals = 0)    
    fpr, tpr, thres_roc = roc_curve(test[lab], pred, pos_label=1)
    prec, rec, thres_pr = precision_recall_curve(test[lab], pred, pos_label= 1)
    test_loss = model.evaluate(test[feat], test[lab], verbose=0)
    
    def model_write(string):
        file.write(string + '\n')

    with open(filename, mode='w') as file:
        model.summary(print_fn= model_write)
        file.write('\n\n')
        for key in args:
            file.write(f'{key}: {args[key]} \n')
        file.write('-'*70 + '\n')
        file.write('Optimizer: {} \n'.format(model.optimizer._name))
        file.write('Loss function: {} \n'.format(model.loss.name))
        file.write('-'*70 + '\n')
        file.write('Loss on test dataset: {:.4g} \n'.format(test_loss))
        file.write('ROC curve AUC: {}\n'.format(auc(fpr, tpr)))
        file.write('Precision-recall AUC: {}\n'.format(auc(rec, prec)))
        file.write('-'*70 + '\n')
        file.write(classification_report(test[lab],pred_classes))

def plot_report(pred, history, test_lab, filename):
    
    # plot loss during training, roc curve, precision-recall curve and confusion matrix to file
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
    
    pred_classes = np.round(pred, decimals = 0)    
    fpr, tpr, thres_roc = roc_curve(test_lab, pred, pos_label=1)
    prec, rec, thres_pr = precision_recall_curve(test_lab, pred, pos_label= 1)
    
    plt.figure(figsize=(10,10))
    
    # loss during training
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend()
    
    # confusion matrix
    plt.subplot(2, 2, 2)
    conf_m = confusion_matrix(test_lab, pred_classes)
    df_conf_m = pd.DataFrame(conf_m, index=[0,1], columns=[0,1])
    sns.heatmap(df_conf_m, cmap=sns.color_palette('light:teal', as_cmap=True), annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # ROC curve
    plt.subplot(2, 2, 3)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid()
    
    # Precision-recall curve
    a = len (test_lab[test_lab == 1])/len(test_lab)
    plt.subplot(2, 2, 4)
    plt.plot(rec, prec)
    plt.plot([0, 1], [a, a] , color='gray', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.xlim([0,1])
    plt.ylim([0,1])

    plt.savefig(filename, dpi=150, bbox_inches= 'tight') 

def test_thresholds(thresholds, pred, test_lab, filename):
    
    from sklearn.metrics import confusion_matrix
    
    # get predictions with different thresholds
    pred_c_thres = []    
    for t in thresholds:
        pred_c_thres.append(
            [math.floor(p[0]) if p[0] < t else math.ceil(p[0]) for p in pred]
        )
    
    # plot conf matrix for each
    plt.figure(figsize=(15,4))
    
    for i,(p,t) in enumerate(zip(pred_c_thres,thresholds)):
        plt.subplot(1, 3, i+1)
        plt.title(f'threshold = {t}')
        c = confusion_matrix(test_lab, p)
        c = pd.DataFrame(c, index=[0,1], columns=[0,1])
        sns.heatmap(c, cmap=sns.color_palette('light:indigo', as_cmap=True), annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
    plt.savefig(filename, dpi=150, bbox_inches= 'tight')

def test_best_thresholds(pred, test_lab, filename):

    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
      
    # With G-Mean
    fpr, tpr, thres_roc = roc_curve(test_lab, pred, pos_label=1)
    gmeans = (tpr * (1-fpr))**(1/2)
    max_gmeans = np.argmax(gmeans) # locate the index of the largest g-mean
    
    # ROC Curve
    plt.figure(figsize=(10,10))
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.plot(fpr[max_gmeans], tpr[max_gmeans], marker='o', color='black')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'G-Mean = {gmeans[max_gmeans]:.3f}')
    plt.grid()
    # Conf matrix
    pred_gmeans = [math.floor(p[0]) if p[0] < thres_roc[max_gmeans] else math.ceil(p[0]) for p in pred]
    plt.subplot(2, 2, 2)
    plt.title(f'Threshold = {thres_roc[max_gmeans] :.3f}')
    c = confusion_matrix(test_lab, pred_gmeans)
    c = pd.DataFrame(c, index=[0,1], columns=[0,1])
    sns.heatmap(c, cmap=sns.color_palette('light:chocolate', as_cmap=True), annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # With F-Score
    prec, rec, thres_pr = precision_recall_curve(test_lab, pred, pos_label= 1)
    fscore = (2 * prec * rec) / (prec + rec)
    max_f = np.argmax(fscore)

    # Precision-recall curve
    plt.subplot(2, 2, 3)
    plt.plot(rec, prec)
    plt.plot(rec[max_f], prec[max_f], marker='o', color='black')
    a = len (test_lab[test_lab == 1])/len(test_lab)
    plt.plot([0, 1], [a, a] , color='gray', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'F-Score = {fscore[max_f] :.3f}')
    plt.grid()
    plt.xlim([0,1])
    plt.ylim([0,1])
    # Conf matrix
    pred_f = [math.floor(p[0]) if p[0] < thres_pr[max_f] else math.ceil(p[0]) for p in pred]
    plt.subplot(2, 2, 4)
    c = confusion_matrix(test_lab, pred_f)
    c = pd.DataFrame(c, index=[0,1], columns=[0,1])
    sns.heatmap(c, cmap=sns.color_palette('light:mediumvioletred', as_cmap=True), annot=True, fmt='d')
    plt.title(f'Threshold = {thres_pr[max_f] :.3f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.savefig(filename, dpi=150, bbox_inches= 'tight')

def class_weights(df, lab):
    # compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    wei =  compute_class_weight(class_weight = 'balanced', classes = np.unique(df[lab]), y = df[lab])
    weights = {}
    for w,l in zip(wei,np.unique(df[lab])):
        weights[l] = w
    return weights

def smote(train, feat, lab):
    # deal with imbalanced data using undersampling + smote
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    # undersampling majority class
    rus = RandomUnderSampler(sampling_strategy= 3./7, replacement= False)
    rus_feat, rus_lab = rus.fit_resample(train[feat], train[lab])
    # smote
    smote = SMOTE(sampling_strategy= 2./3)
    smote_feat, smote_lab = smote.fit_resample(rus_feat, rus_lab)
    
    n_mem = len(smote_lab[smote_lab == 1])
    n_no = len(smote_lab[smote_lab == 0])
    n = len(smote_lab)
    print('SMOTE:')
    print ('Members: {} ({:.2f}%)'.format(n_mem, n_mem/n*100))
    print ('Non members: {} ({:.2f}%)'.format(n_no, n_no/n*100))
    
    smote_feat[lab] = smote_lab
    return smote_feat

 
        
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name') # file with model, given as argument in command line without .py extension
    parser.add_argument('-e', '--model_exists', action='store_true') # does the model already exist?
    parser.add_argument('-f','--fields_list', nargs='+', action='store', default=['W01','W02','W03', 'W04']) # list of HSC fields
    parser.add_argument('-r', '--random_state', action='store', default=42, type= int) # for train-val-test split
    parser.add_argument('-t', '--thresholds', action='store_true') # test different thresholds?
    parser.add_argument('-bt', '--best_thresholds', action='store_true') # check best thresholds?
    parser.add_argument('--min_n500', action='store', default=None, type= float) # minimum for cluster n500?
    parser.add_argument('--max_n500', action='store', default=None, type= float) # maximum for cluster n500?
    parser.add_argument('--min_z', action='store', default=None, type= float) # minimum for cluster z?
    parser.add_argument('--max_z', action='store', default=None, type= float) # maximum for cluster z?
    parser.add_argument('--feat_max', action='append', default=[], nargs='+') #limit feature to max. value? Use as --feat_max feature value
    parser.add_argument('--feat_min', action='append', default=[], nargs='+') #limit feature to min. value? Use as --feat_min feature value
    
    model_name = parser.parse_args().model_name
    model_exists = parser.parse_args().model_exists  
    fields_list = parser.parse_args().fields_list  
    random_state = parser.parse_args().random_state
    thresholds = parser.parse_args().thresholds 
    best_thresholds = parser.parse_args().best_thresholds 
    min_n500 = parser.parse_args().min_n500
    max_n500 = parser.parse_args().max_n500
    min_z = parser.parse_args().min_z
    max_z = parser.parse_args().max_z
    feat_max = dict(parser.parse_args().feat_max)
    feat_min = dict(parser.parse_args().feat_min)
    
    # prepare data
    data = read_data(fields_list)
    if any(val != None for val in [min_n500, max_n500, min_z, max_z]):
        data = z_n500_limits(data, min_z, max_z, min_n500, max_n500) 
    features,label = features_labels(df=data)
    if any([feat_max, feat_min]):
        data = feature_limits(data, feat_max, feat_min)
    training,validation,testing = split(df=data, lab=label, ran_state=random_state)
    
    
    # if the model has already been trained and saved, skip the following    
    if not model_exists:
        
        # import variables from file, 
        import importlib
        mod = importlib.import_module(model_name)
        layers = getattr(mod, 'layers')
        compile_params = getattr(mod, 'compile_params')
        epochs = getattr(mod, 'epochs')
        normalization = getattr(mod, 'normalization')
        balance = getattr(mod, 'balance') 
        
        # left model checkpoint to save best model
        callbacks = [
            # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            tf.keras.callbacks.CSVLogger(filename = f'saved_models/{model_name}_log.csv'),
            tf.keras.callbacks.ModelCheckpoint(filepath = f'saved_models/{model_name}.h5', monitor = 'val_loss',  save_best_only = True),
            # tf.keras.callbacks.TensorBoard()
            ]
       
        #include normalization layer?
        if normalization:
            norm = tf.keras.layers.Normalization(input_shape=(len(features),))
            norm.adapt(data = training[features].values)
            layers.insert(0, norm)
        
        # make model and compile. Make dict with arguments for fit function.
        nn_model = tf.keras.Sequential(layers)  
        nn_model.compile(**compile_params)
        fit_params = dict(
            x = training[features].values, 
            y = training[label].values,
            verbose = 2, 
            callbacks = callbacks,
            validation_data = (validation[features].values, validation[label].values), 
            epochs = epochs,
            batch_size = 4096
            )
        
        # class imbalance?
        match balance:
            case None:
                nn_model.fit(**fit_params)
            case 'weights':
                fit_params['class_weight'] = class_weights(data, label)
                nn_model.fit(**fit_params) 
            case 'SMOTE':
                training = smote(training, features, label)
                fit_params['class_weight'] = class_weights(training, label)
                nn_model.fit(**fit_params)
            case _:
                raise ValueError("Invalid value for 'balance'")
                 
    # load best model and history from files
    nn_model = tf.keras.models.load_model(f'saved_models/{model_name}.h5')
    history = pd.read_csv(f'saved_models/{model_name}_log.csv')
    
    # save metrics
    predictions = nn_model.predict(testing[features].values, verbose = 0)
    write_report(predictions, nn_model, testing, features, label, f'metrics/{model_name}.txt', parser.parse_args().__dict__)
    plot_report(predictions, history, testing[label], f'metrics/{model_name}.png')
    
    # test different thresholds
    if thresholds:
        test_thresholds([0.45,0.5,0.55], predictions, testing[label], f'metrics/{model_name}_thresholds.png')
    
    if best_thresholds:
        test_best_thresholds(predictions, testing[label], f'metrics/{model_name}_best-thresholds_.png')
    