"""
Test other feature sets with a 512x512 network with dropout (0.2).
"""

import sys
import tensorflow as tf

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

# Load and prepare data data 
data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', balance= 'weights')
data.main()

features = ['all_features', 'all_features_z_mass', 'all_features_abs_mags', 'all_features_sigma_5', 'all_features_bcg', 'all_features_sigmas']
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
    mod = NNModelController(data = data.copy(), name = feat, layers = layers.copy(), compile_params= compile_params, epochs = 1000)
    mod.main(prep_data= True)