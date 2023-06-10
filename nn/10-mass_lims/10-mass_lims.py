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
mass_lims = [None, 10.5]
names = ['max_mass_none', 'max_mass_10.5']

# Train and test each model
for m,nam in zip(mass_lims, names):

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

    if m:
        data.feat_max = {"log_st_mass": m}

    mod = NNModelController(data = data.copy(), name = nam, layers = layers.copy(), compile_params= compile_params)
    mod.main(prep_data= True)

