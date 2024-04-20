import csv
import sys
import numpy as np

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', balance= 'weights')
data.read_data()

feat_lims = dict(
    gr_abs = [0, 0.25, 0.5, 0.75, 1., 1.25, 1.5, 1.75],
    ri_abs = [-0.25, 0, 0.25, 0.5, 0.75, 1., 1.25],
    iz_abs = [-0.25, 0, 0.25, 0.5, 0.75],
    zy_abs = [-0.5, -0.25, 0, 0.25, 0.5, 0.75],
)

features = ['all_features', 'all_features_bcg', 'all_features_sigmas']

for fl in feat_lims.keys():

    metrics_1 = []

    lims = feat_lims[fl]
    for i in range(len(lims) - 1):
        
        d = data.copy()
        if lims[i+1]:
            d.feat_max = {f"{fl}": lims[i+1]}
        if lims[i]:
            d.feat_min = {f"{fl}": lims[i]}
        d.prep()

        met1 = []

        for feat in features:

            d.features_txt = feat + '.txt'
            d.features_labels()

            mod = NNModelController(data = d, name = feat)
            mod.main(model_exists= True, test_name= f'{fl}/{feat}/{lims[i]}_to_{lims[i+1]}', importances = [])
            met1.append([mod.tester.p, mod.tester.r, mod.tester.specificity, mod.tester.accuracy, mod.tester.f1, mod.tester.pr_auc, mod.tester.roc_auc, mod.tester.threshold])

        metrics_1.append(met1)

    metrics_1 = np.array(metrics_1)

    with open(f'metrics_{fl}.csv', 'w', newline= '') as file:
        writer = csv.writer(file)
        writer.writerow(['Features', 'min', 'max', 'Precision', 'Recall', 'Specificity', 'Accuracy', 'F1-score', 'PR AUC', 'ROC AUC', 'Threshold'])

        for i,feat in enumerate(features):
            for i,met1 in enumerate(zip(metrics_1[:,i])):
                writer.writerow([feat, lims[i], lims[i+1]]+['{:.4f}'.format(m) for m in met1[0]])