import sys
import tensorflow as tf

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

# Load and prepare data data 
data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', fields_list=['W01', 'W02'], balance= 'weights')
data.main()

features = ['all_features.txt', 'features1.txt', 'all_features_bcg.txt', 'all_features_z_mass.txt']
names = ['all_features', 'features1', 'all_features_bcg', 'all_features_z_mass']

for feat,nam in zip(features,names):

    # Architecture
    layers = [
        tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    compile_params = dict(
        optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0005),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics=[]   
    )

    data.features_txt = feat
    mod = NNModelController(data = data.copy(), name = nam, layers = layers.copy(), compile_params= compile_params)
    mod.main(prep_data= True)