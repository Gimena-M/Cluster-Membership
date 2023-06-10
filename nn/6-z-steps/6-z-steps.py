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
z_lims = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

# Train and test each model
for i in range(len(z_lims) - 1):
    
    data.min_z = z_lims[i]
    data.max_z = z_lims[i + 1]
    name = f"z_{z_lims[i]}_to_{z_lims[i+1]}"

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

    mod = NNModelController(data = data.copy(), name = name, layers = layers.copy(), compile_params= compile_params)
    mod.main(prep_data= True)

