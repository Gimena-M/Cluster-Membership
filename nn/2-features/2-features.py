"""
Test other feature sets with a 512x512 network with dropout (0.2).
"""

import sys
import tensorflow as tf
import csv

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

# Load and prepare data data 
data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', balance= 'weights')
data.main()

metrics = []
# features = ['all_features_bcg', 'all_features_sigmas'] #'all_features', 'all_features_z_mass', 'all_features_abs_mags', 
features = ['all_features']
for feat in features: 

    layers = [
        tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    
    if ("bcg" in feat)|("sigmas" in feat):
        learning_rate = 1e-4
    else:
        learning_rate = 1e-5

    compile_params = dict(
        optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics=[]   
    )

    data.features_txt = feat + '.txt'
    data.features_labels()
    mod = NNModelController(data = data.copy(), name = feat, layers = layers.copy(), compile_params= compile_params, epochs = 1000)
    mod.main(model_exists= True, permutation_train_max_samples = 300_000, permutation_test_max_samples = 300_000)

    # save metrics in nested list
    metrics.append([mod.tester.p, mod.tester.r, mod.tester.specificity, mod.tester.accuracy, mod.tester.f1, mod.tester.pr_auc, mod.tester.roc_auc, mod.tester.threshold])

# write metrics to csv
    
with open('metrics.csv', 'w', newline= '') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'Precision', 'Recall', 'Specificity', 'Accuracy', 'F1-score', 'PR AUC', 'ROC AUC', 'Threshold'])
    for nam,met in zip(features, metrics):
        writer.writerow([nam]+['{:.4f}'.format(m) for m in met])