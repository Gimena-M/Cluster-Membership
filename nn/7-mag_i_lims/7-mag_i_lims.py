import sys
import numpy as np
import tensorflow as tf

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

# Load and prepare data data 
# data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', fields_list=['W02', 'W03', 'W04'], balance= 'weights')
# data.main()

# # Maximum values for i
# i_lims = [None, 21, 19]
# names = ['max_i_' + n for n in ['none','21','19']]

# # Train and test each model
# for i,nam in zip(i_lims, names):

#     if i:
#         data.feat_max = {"i_cmodel_mag": i}

#     # Architecture
#     layers = [
#         tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#         ]
#     compile_params = dict(
#         optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0005),
#         loss = tf.keras.losses.BinaryCrossentropy(),
#         metrics=[]   
#     )
    
#     mod = NNModelController(data = data.copy(), name = nam, layers = layers.copy(), compile_params= compile_params)
#     mod.main(prep_data= True)







# now with quantiles

data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', fields_list=['W02', 'W03', 'W04'], balance= 'weights')
data.main()

q1 = np.quantile(data.data['i_cmodel_mag'], 0.25)
q3 = np.quantile(data.data['i_cmodel_mag'], 0.75)

lims = [None, None, q1, q3, None]
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
    if lims[i+1]:
        d.feat_max = {"i_cmodel_mag": lims[i+1]}
    if lims[i]:
        d.feat_min = {"i_cmodel_mag": lims[i]}
    
    mod = NNModelController(data = d, name = nam, layers = layers.copy(), compile_params= compile_params)
    mod.main(prep_data= True)
