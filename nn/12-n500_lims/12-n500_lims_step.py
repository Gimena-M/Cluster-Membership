import csv
import sys
import numpy as np

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', balance= 'weights')
data.read_data()

lims = [0, 10, 20, 30, 40, 50, 60, 999]
features = ['all_features', 'all_features_bcg', 'all_features_sigmas']

metrics_1 = []
metrics_2 = []

for i in range(len(lims) - 1):
    
    d = data.copy()
    d.max_n500 = lims[i+1]
    d.min_n500 = lims[i]
    d.prep()

    met1 = []
    met2 = []

    for feat in features:

        print(f'Testing {feat} for {lims[i]} < N500 < {lims[i+1]}')

        d.features_txt = feat + '.txt'
        d.features_labels()

        mod = NNModelController(data = d, name = feat)
        mod.main(model_exists= True, test_name= f'step/{feat}/{lims[i]}_to_{lims[i+1]}', importances = None)
        met1.append([mod.tester.p, mod.tester.r, mod.tester.specificity, mod.tester.accuracy, mod.tester.f1, mod.tester.pr_auc, mod.tester.roc_auc, mod.tester.threshold])

        mod.main(model_exists= True, resume_training= True, test_name= f'step/{feat}/{lims[i]}_to_{lims[i+1]}--retrained', retrain_name= f'step/{feat}/{lims[i]}_to_{lims[i+1]}--retrained', importances = None)
        met2.append([mod.tester.p, mod.tester.r, mod.tester.specificity, mod.tester.accuracy, mod.tester.f1, mod.tester.pr_auc, mod.tester.roc_auc, mod.tester.threshold])

    metrics_1.append(met1)
    metrics_2.append(met2)

metrics_1 = np.array(metrics_1)
metrics_2 = np.array(metrics_2)

with open('metrics_step.csv', 'w', newline= '') as file:
    writer = csv.writer(file)
    writer.writerow(['Features', 'n_min', 'n_max', 'Precision', 'Recall', 'Specificity', 'Accuracy', 'F1-score', 'PR AUC', 'ROC AUC', 'Threshold', 'Precision (Ret)', 'Recall (Ret)', 'Specificity (Ret)', 'Accuracy (Ret)', 'F1-score (Ret)', 'PR AUC (Ret)', 'ROC AUC (Ret)', 'Threshold (Ret)'])

    lims = [f'{i :.4f}'if i else 'none' for i in lims ]

    for i,feat in enumerate(features):
        for i,(met1,met2) in enumerate(zip(metrics_1[:,i], metrics_2[:,i])):
            writer.writerow([feat, lims[i], lims[i+1]]+['{:.4f}'.format(m) for m in met1]+['{:.4f}'.format(m) for m in met2])