import sys
import tensorflow as tf

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

# Load and prepare data data 
data_1 = DataHandler(validation_sample= True, features_txt= 'all_features.txt', fields_list=['W01', 'W02'], balance= 'weights')
data_2 = DataHandler(validation_sample= True, features_txt= 'all_features.txt', fields_list=['W01', 'W02'], balance= 'weights', z_filtered= True)

data = [data_1, data_2]
names = ["all", "z_filtered"]

for dat,nam in zip(data,names):
    
    # Architecture
    layers = [
        tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    # Compile parameters:
    compile_params = dict(
        optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0005),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics=[]   
    )

    mod = NNModelController(compile_params= compile_params, data= dat, name= nam, layers= layers)
    mod.main(read_data= True)