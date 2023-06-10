import sys
import tensorflow as tf

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

# Load and prepare data data 
data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', fields_list=['W01', 'W02', 'W03', 'W04'], balance= 'weights')
data.main()

# Maximum values for z
z_lims = [None, 1.0, 0.7, 0.5, 0.3]
names = ['max_z_' + n for n in ['none','1.0','0.7','0.5','0.3']]

# Train and test each model
for z,nam in zip(z_lims, names):

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

    data.max_z = z
    mod = NNModelController(data = data, name = nam, layers = layers.copy(), compile_params= compile_params)
    mod.main(prep_data= True)

