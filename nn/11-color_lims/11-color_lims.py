import sys
import numpy as np
import tensorflow as tf

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

# Load and prepare data data 
data = DataHandler(validation_sample= True, features_txt= 'all_features.txt', fields_list=['W02', 'W03', 'W04'], balance= 'weights')
data.main()

q1 = np.quantile(data.data['ri'], 0.25)
q3 = np.quantile(data.data['ri'], 0.75)

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
        d.feat_max = {"ri": lims[i+1]}
    if lims[i]:
        d.feat_min = {"ri": lims[i]}
    
    mod = NNModelController(data = d, name = nam, layers = layers.copy(), compile_params= compile_params)
    mod.main(prep_data= True)




# # Test with no limits, with g-r or r-i smaller than 1., and with g-r or r-i larger than 1.
# gr_lim = 1.
# ri_lim = 1.

# def func():
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
    
#     mod = NNModelController(data = data.copy(), name = name, layers = layers.copy(), compile_params= compile_params)
#     mod.main(prep_data= True)

# # No limits
# data.feat_max = {}
# data.feat_min = {}
# name = "no_lim"
# func()

# # g - r >= 1.
# data.feat_max = {}
# data.feat_min = {"gr": gr_lim}
# name = "gr_larger_than_1"
# func()


# # g - r <= 1.
# data.feat_max = {"gr": gr_lim}
# data.feat_min = {}
# name = "gr_smaller_than_1"
# func()


# # r - i >= 1.
# data.feat_max = {}
# data.feat_min = {"ri": ri_lim}
# name = "ri_larger_than_1"
# func()


# # r - i <= 1.
# data.feat_max = {"ri": ri_lim}
# data.feat_min = {}
# name = "ri_smaller_than_1"
# func()