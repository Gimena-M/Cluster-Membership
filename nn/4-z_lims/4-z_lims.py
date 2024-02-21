"""
Test models (photometric features, sigmas, and BCG's z) with dataset limited in redshift.
Test performance before training, retrain, and test again.
"""

import csv
import sys
import numpy as np

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

# Load and prepare data data 
data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', balance= 'weights')
data.main()

# Maximum values for z
z_lims = [None, 1.0, 0.7, 0.5, 0.3]
z_names = ['--max_z_' + n for n in ['none', '1.0','0.7','0.5','0.3']]

# Feature sets
features = ['all_features', ] # 'all_features_bcg', 'all_features_sigmas'
metrics_1 = []
metrics_2= []

# Train and test each model
for z,znam in zip(z_lims, z_names):
    
    data.max_z = z
    met1 = []
    met2 = []

    for feat in features:
        data.features_txt = feat + '.txt'

        # Test before retraining
        name = f'{feat}/{feat}{znam}'
        print(f'Testing {name}', flush = True)
        mod = NNModelController(data = data, name = feat)
        mod.main(prep_data= True, model_exists= True, test_name= name)
        met1.append([mod.tester.p, mod.tester.r, mod.tester.specificity, mod.tester.accuracy, mod.tester.f1, mod.tester.pr_auc, mod.tester.roc_auc, mod.tester.threshold])

        if ('none' not in name):
            # Retrain
            name = f'{feat}/{feat}{znam}--retrained'
            print(f'Training {name}', flush= True)
            mod.main(model_exists= True, resume_training= True, test_name= name, retrain_name= name)
            met2.append([mod.tester.p, mod.tester.r, mod.tester.specificity, mod.tester.accuracy, mod.tester.f1, mod.tester.pr_auc, mod.tester.roc_auc, mod.tester.threshold])
            # mod2 = NNModelController(data = data, name = name)
            # mod2.main(model_exists=True)
            # met2.append([mod2.tester.p, mod2.tester.r, mod2.tester.specificity, mod2.tester.accuracy, mod2.tester.f1, mod2.tester.pr_auc, mod2.tester.roc_auc, mod2.tester.threshold])
        else:
            met2.append([0.]*8)

    metrics_1.append(met1)
    metrics_2.append(met2)

metrics_1 = np.array(metrics_1)
metrics_2 = np.array(metrics_2)

with open('metrics.csv', 'w', newline= '') as file:
    writer = csv.writer(file)
    writer.writerow(['Features', 'Z', 'Precision', 'Recall', 'Specificity', 'Accuracy', 'F1-score', 'PR AUC', 'ROC AUC', 'Threshold', 'Precision (Ret)', 'Recall (Ret)', 'Specificity (Ret)', 'Accuracy (Ret)', 'F1-score (Ret)', 'PR AUC (Ret)', 'ROC AUC (Ret)', 'Threshold (Ret)'])

    for i,feat in enumerate(features):
        for znam,met1,met2 in zip(z_names, metrics_1[:,i], metrics_2[:,i]):
            writer.writerow([feat,znam]+['{:.4f}'.format(m) for m in met1]+['{:.4f}'.format(m) for m in met2])



    
    

