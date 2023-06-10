import sys
import tensorflow as tf

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

# Load and prepare data data 
data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', fields_list=['W01', 'W02', 'W03', 'W04'], balance= 'weights')
data.main()

# Maximum values for i
i_lims = [None, 21, 19]
names = ['max_i_' + n for n in ['none','21','19']]

# Train and test each model
for i,nam in zip(i_lims, names):

    if i:
        data.feat_max = {"i_cmodel_mag": i}

    # Architecture
    layers = [
        tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    compile_params = dict(
        optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0005),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics=[]   
    )
    
    mod = NNModelController(data = data.copy(), name = nam, layers = layers.copy(), compile_params= compile_params)
    mod.main()
