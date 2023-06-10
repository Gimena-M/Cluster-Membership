import sys
import tensorflow as tf

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

# Load and prepare data data 
data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', fields_list=['W01', 'W02', 'W03', 'W04'], balance= 'weights')
data.main()

# Lists of architectures to try
layers_l = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]
layers_m = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]
layers_s = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]

layers_l_dropout = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]
layers_m_dropout = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]
layers_s_dropout = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]

layers_l_gaussian = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.GaussianDropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.GaussianDropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.GaussianDropout(0.2),
    tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
    tf.keras.layers.GaussianDropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]
layers_m_gaussian = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.GaussianDropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.GaussianDropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.GaussianDropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]
layers_s_gaussian = [
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.GaussianDropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.GaussianDropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]

layers = [layers_l, layers_m, layers_s, layers_l_dropout, layers_m_dropout, layers_s_dropout, layers_l_gaussian, layers_m_gaussian, layers_s_gaussian]
names = ['l', 'm', 's', 'l_dropout', 'm_dropout', 's_dropout', 'l_gaussian', 'm_gaussian', 's_gaussian']

# Train and test each model
for nam,lay in zip(names,layers):

    # Compile parameters:
    compile_params = dict(
        optimizer = tf.keras.optimizers.Adam(),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics=[]   
    )

    mod = NNModelController(layers= lay, name= nam, data= data, compile_params= compile_params)
    mod.main(read_data= False, prep_data= False)

