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

q1 = np.quantile(data.data['ri'], 0.25)
q3 = np.quantile(data.data['ri'], 0.75)

lims = [None, None, q1, q3, None]
names = ['all', 'less_than_q1', 'between_q1_q3', 'more_than_q3']
features = ['all_features', 'all_features_bcg', 'all_features_sigmas']

metrics_1 = []
metrics_2 = []

# Train and test each model
for i,nam in enumerate(names):

    d = data.copy()
    if lims[i+1]:
        d.feat_max = {"ri": lims[i+1]}
    if lims[i]:
        d.feat_min = {"ri": lims[i]}
    d.prep()

    met1 = []
    met2 = []

    for feat in features:

        d.features_txt = feat + '.txt'
        d.features_labels()

        mod = NNModelController(data = d, name = feat)
        mod.main(model_exists= True, test_name= f'quantile/{feat}/{nam}')
        met1.append([mod.tester.p, mod.tester.r, mod.tester.specificity, mod.tester.accuracy, mod.tester.f1, mod.tester.pr_auc, mod.tester.roc_auc, mod.tester.threshold])

        if nam != 'all':
            mod.main(model_exists= True, resume_training= True, test_name= f'quantile/{feat}/{nam}--retrained', retrain_name= f'quantile/{feat}/{nam}--retrained')
        met2.append([mod.tester.p, mod.tester.r, mod.tester.specificity, mod.tester.accuracy, mod.tester.f1, mod.tester.pr_auc, mod.tester.roc_auc, mod.tester.threshold])

    metrics_1.append(met1)
    metrics_2.append(met2)

metrics_1 = np.array(metrics_1)
metrics_2 = np.array(metrics_2)

with open('metrics.csv', 'w', newline= '') as file:
    writer = csv.writer(file)
    writer.writerow(['Features', 'ri_min', 'ri_max', 'Precision', 'Recall', 'Specificity', 'Accuracy', 'F1-score', 'PR AUC', 'ROC AUC', 'Threshold', 'Precision (Ret)', 'Recall (Ret)', 'Specificity (Ret)', 'Accuracy (Ret)', 'F1-score (Ret)', 'PR AUC (Ret)', 'ROC AUC (Ret)', 'Threshold (Ret)'])

    lims = [f'{i :.4f}'if i else 'none' for i in lims ]

    for i,feat in enumerate(features):
        for i,(met1,met2) in enumerate(zip(metrics_1[:,i], metrics_2[:,i])):
            writer.writerow([feat, lims[i], lims[i+1]]+['{:.4f}'.format(m) for m in met1]+['{:.4f}'.format(m) for m in met2])

