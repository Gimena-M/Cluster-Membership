import sys
import tensorflow as tf

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

# Load and prepare data data 
data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', fields_list=['W01', 'W02', 'W03', 'W04'], balance= 'weights')
data.main()



name = 'm_adam_0.001'
layers = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]
compile_params = dict(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=[]   
)
mod = NNModelController(data = data, name = name, layers= layers, compile_params= compile_params)
mod.main()



name = 's_adam_0.001'
layers = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]
compile_params = dict(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=[]   
)
mod = NNModelController(data = data, name = name, layers= layers, compile_params= compile_params)
mod.main()


name = 'm_adam_0.0005'
layers = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
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
mod = NNModelController(data = data, name = name, layers= layers, compile_params= compile_params)
mod.main()





name = 's_adam_0.0005'
layers = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]
compile_params = dict(
    optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0005),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=[]   
)
mod = NNModelController(data = data, name = name, layers= layers, compile_params= compile_params)
mod.main()




name = 'm_adamw_0.001'
layers = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]
compile_params = dict(
    optimizer = tf.keras.optimizers.experimental.AdamW(),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=[]   
)
mod = NNModelController(data = data, name = name, layers= layers, compile_params= compile_params)
mod.main()





name = 's_adamw_0.001'
layers = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]
compile_params = dict(
    optimizer = tf.keras.optimizers.experimental.AdamW(),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=[]   
)
mod = NNModelController(data = data, name = name, layers= layers, compile_params= compile_params)
mod.main()


