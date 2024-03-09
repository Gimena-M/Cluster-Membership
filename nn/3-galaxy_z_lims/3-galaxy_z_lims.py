import csv
import sys
import numpy as np

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', balance= 'weights')
data.read_data()

lims = [None, 1.2, 0.9, 0.6, 0.3]
names = ['all', 'z_max_1.2', 'z_max_0.9', 'z_max_0.6', 'z_max_0.3']
features = ['all_features', 'all_features_bcg', 'all_features_sigmas']

metrics_1 = []
metrics_2 = []

for lim,nam in zip(lims, names):

    if lim:
        data.feat_max = {"phot_z": lim}
    data.prep()

    met1 = []
    met2 = []

    for feat in features:

        data.features_txt = feat + '.txt'
        data.features_labels()

        mod = NNModelController(data = data, name = feat)
        mod.main(model_exists= True, test_name= f'{feat}/{nam}')
        met1.append([mod.tester.p, mod.tester.r, mod.tester.specificity, mod.tester.accuracy, mod.tester.f1, mod.tester.pr_auc, mod.tester.roc_auc, mod.tester.threshold])

        if nam != 'all':
            mod.main(model_exists= True, resume_training= True, test_name= f'{feat}/{nam}--retrained', retrain_name= f'{feat}/{nam}--retrained')
        met2.append([mod.tester.p, mod.tester.r, mod.tester.specificity, mod.tester.accuracy, mod.tester.f1, mod.tester.pr_auc, mod.tester.roc_auc, mod.tester.threshold])

    metrics_1.append(met1)
    metrics_2.append(met2)

metrics_1 = np.array(metrics_1)
metrics_2 = np.array(metrics_2)

with open('metrics.csv', 'w', newline= '') as file:
    writer = csv.writer(file)
    writer.writerow(['Features', 'z_max', 'Precision', 'Recall', 'Specificity', 'Accuracy', 'F1-score', 'PR AUC', 'ROC AUC', 'Threshold', 'Precision (Ret)', 'Recall (Ret)', 'Specificity (Ret)', 'Accuracy (Ret)', 'F1-score (Ret)', 'PR AUC (Ret)', 'ROC AUC (Ret)', 'Threshold (Ret)'])

    lims = [f'{i :.4f}'if i else 'none' for i in lims ]

    for i,feat in enumerate(features):
        for i,(met1,met2) in enumerate(zip(metrics_1[:,i], metrics_2[:,i])):
            writer.writerow([feat, lims[i]]+['{:.4f}'.format(m) for m in met1]+['{:.4f}'.format(m) for m in met2])