"""
Test performance for galaxies in z ranges with a 0.1 step, retrain and test again.
"""

import csv
import sys
import numpy as np

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

# Load and prepare data data 
data_ = DataHandler(validation_sample= True, features_txt= 'all_features.txt', balance= 'weights')
data_.read_data()

# Maximum values for z
z_lims = [0.2,0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
features = ['all_features', 'all_features_bcg', 'all_features_sigmas']

metrics_1 = []
metrics_2 = []

# Train and test each model
for i in range(len(z_lims) - 1):
    
    data = data_.copy()
    data.feat_max = {"phot_z": z_lims[i+1]}
    data.feat_min = {"phot_z": z_lims[i]}
    data.prep()
    name = f"z_{z_lims[i]}_to_{z_lims[i+1]}"

    met1 = []
    met2 = []

    for feat in features:

        data.features_txt = feat + '.txt'
        data.features_labels()

        print(f'Testing {feat} {name}', flush = True)
        mod = NNModelController(data = data.copy(), name = feat)
        mod.main(model_exists= True, test_name= f'{feat}/{name}', permutation_train_max_samples= 100_000 , permutation_test_max_samples= 15_000)
        met1.append([mod.tester.p, mod.tester.r, mod.tester.specificity, mod.tester.accuracy, mod.tester.f1, mod.tester.pr_auc, mod.tester.roc_auc, mod.tester.threshold])

        # Retrain
        print(f'Training {feat} {name}', flush= True)
        mod.main(model_exists= True, resume_training= True, test_name= f'{feat}/{name}--retrained', retrain_name= f'{feat}/{name}--retrained', importances= None)
        met2.append([mod.tester.p, mod.tester.r, mod.tester.specificity, mod.tester.accuracy, mod.tester.f1, mod.tester.pr_auc, mod.tester.roc_auc, mod.tester.threshold])

    metrics_1.append(met1)
    metrics_2.append(met2)

metrics_1 = np.array(metrics_1)
metrics_2 = np.array(metrics_2)

with open('metrics.csv', 'w', newline= '') as file:
    writer = csv.writer(file)
    writer.writerow(['Features', 'z_min', 'z_max', 'Precision', 'Recall', 'Specificity', 'Accuracy', 'F1-score', 'PR AUC', 'ROC AUC', 'Threshold', 'Precision (Ret)', 'Recall (Ret)', 'Specificity (Ret)', 'Accuracy (Ret)', 'F1-score (Ret)', 'PR AUC (Ret)', 'ROC AUC (Ret)', 'Threshold (Ret)'])

    for i,feat in enumerate(features):
        for i,(met1,met2) in enumerate(zip(metrics_1[:,i], metrics_2[:,i])):
            writer.writerow([feat, z_lims[i], z_lims[i+1]]+['{:.4f}'.format(m) for m in met1]+['{:.4f}'.format(m) for m in met2])