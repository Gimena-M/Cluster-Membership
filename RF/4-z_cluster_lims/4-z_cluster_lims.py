import csv
import sys
import numpy as np

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from RFModelController import RFModelController

# Load and prepare data data 
data = DataHandler(validation_sample= False, features_txt= 'all_features.txt', balance= 'weights')
data.read_data()

# Maximum values for z
z_lims = [1.0, 0.7, 0.5, 0.3] #None
z_names = ['--max_z_' + n for n in ['1.0','0.7','0.5','0.3']] #'none', 

# Feature sets
features = ['all_features', 'all_features_bcg', 'all_features_sigmas']
metrics_1 = []

# Train and test each model
for z,znam in zip(z_lims, z_names):
    
    data.max_z = z
    data.prep()
    met1 = []

    # permutation importance: use only up to 200_000 samples
    if data.testing.shape[0] > 200_000:
        max_perm_test = 200_000
    else:
        max_perm_test = 1.0
    if data.training.shape[0] > 200_000:
        max_perm_train = 200_000
    else:
        max_perm_train = 1.0

    for feat in features:
        data.features_txt = feat + '.txt'
        data.features_labels()

        # Test before retraining
        name = f'{feat}/{feat}{znam}'
        print(f'Testing {name}', flush = True)
        mod = RFModelController(data = data, name = feat)

        mod.main_model(model_exists= True, prep_data= True, test_name= name, permutation_test_max_samples= max_perm_test, permutation_train_max_samples= max_perm_train)
        met1.append([mod.tester.p, mod.tester.r, mod.tester.specificity, mod.tester.accuracy, mod.tester.f1, mod.tester.pr_auc, mod.tester.roc_auc, mod.tester.log_loss, mod.tester.threshold])

    metrics_1.append(met1)
metrics_1 = np.array(metrics_1)

with open('metrics.csv', 'w', newline= '') as file:
    writer = csv.writer(file)
    writer.writerow(['Features', 'Z', 'Precision', 'Recall', 'Specificity', 'Accuracy', 'F1-score', 'PR AUC', 'ROC AUC', 'Log loss', 'Threshold'])

    for i,feat in enumerate(features):
        for znam,met1,met2 in zip(z_names, metrics_1[:,i]):
            writer.writerow([feat,znam]+['{:.4f}'.format(m) for m in met1])



    
    

