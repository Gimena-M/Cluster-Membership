import sys
import tensorflow as tf
import numpy as np

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

# Load and prepare data data 
data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', fields_list=['W02', 'W03', 'W04'], balance= 'weights')
data.main()

q1 = np.quantile(data.data['i_cmodel_mag_abs'], 0.25)
q3 = np.quantile(data.data['i_cmodel_mag_abs'], 0.75)

mag_lims = [None, None, q1, q3, None]
names = ['all', 'less_than_q1', 'between_q1_q3', 'more_than_q3']

# Train and test each model
for i,nam in enumerate(names):

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

    d = data.copy()
    if mag_lims[i+1]:
        d.feat_max = {"i_cmodel_mag_abs": mag_lims[i+1]}
    if mag_lims[i]:
        d.feat_min = {"i_cmodel_mag_abs": mag_lims[i]}
    
    mod = NNModelController(data = d, name = nam, layers = layers.copy(), compile_params= compile_params)
    mod.main(prep_data= True)

