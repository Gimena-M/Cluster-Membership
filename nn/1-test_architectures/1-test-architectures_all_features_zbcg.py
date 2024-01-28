import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import product
import random

# Imports from Classes directory
sys.path.append('../../Classes')
from DataHandler import DataHandler
from NNModelController import NNModelController

# final model

data = DataHandler(validation_sample= True, features_txt= 'all_features_bcg.txt', balance= 'weights')
data.main()

layers = [
    tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]

compile_params = dict(
    optimizer = tf.keras.optimizers.Adam(learning_rate= 1e-4),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=[]   
)

name = 'all_features+bcg'

mod = NNModelController(layers= layers, name= name, data= data, compile_params= compile_params, epochs= 1000, batch_size= 2000)
mod.main(read_data= False, prep_data= False)


#-----------------------------
# Architecture searches

# Load and prepare data data 
data = DataHandler(validation_sample= True, features_txt= 'all_features_bcg.txt', fields_list=['W02'], balance= 'weights')
data.main()



# Architecture parameters:

# 1st run
# archs = dict(
#     n_layers = [2,3,4], 
#     n_units = [32,64,128,256,512],
#     dropout = [0, 0.25],
# )
# learning_rate = '1e-4'

# names = []
# losses = []
# pr_auc = []
# roc_auc = []
# f1s = []


# for n_lay in archs['n_layers']:

#     # possible architectures for this n_layers
#     combinations_lay = list(product(*[archs['n_units']]*n_lay))
#     combinations_drop = list(product(*[archs['dropout']]*n_lay))

#     combinations = list(product(combinations_lay,combinations_drop))
   
#     # pick randomly only 20 combinations
#     if len(combinations) > 20: combinations = random.sample(combinations, 20)
    
#     for comb in combinations:

#         layers = []
#         name_layers = ''
#         name_dropout = ''
#         for unit,drop in zip(comb[0],comb[1]):
#             layers.append(tf.keras.layers.Dense(unit, activation=tf.keras.activations.relu))
#             name_layers = name_layers + f'{unit}x'
            
#             # add dropout layer?
#             if drop:
#                 layers.append(tf.keras.layers.Dropout(drop))
            
#             name_dropout = name_dropout + f'{drop :.1f}+'
                    
#         layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
#         name = str(n_lay) + f'__{learning_rate}__' + name_layers[:-1] + '___' + name_dropout[:-1]

#         compile_params = dict(
#             optimizer = tf.keras.optimizers.Adam(learning_rate= float(learning_rate)),
#             loss = tf.keras.losses.BinaryCrossentropy(),
#             metrics=[]   
#         )
        
#         print ('-'*70)
#         print(f'MODEL: {name}')
#         mod = NNModelController(layers= layers, name= 'bcg_run1/' + name, data= data, compile_params= compile_params, epochs= 1000, batch_size= 2000)
#         mod.main(read_data= False, prep_data= False)

#         names.append(name)
#         losses.append(mod.tester.test_loss)
#         pr_auc.append(mod.tester.pr_auc)
#         roc_auc.append(mod.tester.roc_auc)
#         f1s.append(mod.tester.f1)
        
# df = pd.DataFrame(dict(model = names, loss = losses, pr_auc = pr_auc, roc_auc = roc_auc, f1 = f1s))
# df.to_csv('bcg_search_summary_1.csv', index= False, mode= 'a')


# 2nd run
# networks with at least one 512 neuron layer perform better
# networks with dropout perform better
# archs = dict(
#     n_layers = [4], 
#     n_units = [32, 64, 128,256,512],
#     dropout = [0.2, 0.5],
# )
# learning_rate = '1e-4'


# for n_lay in archs['n_layers']:

#     names = []
#     losses = []
#     pr_auc = []
#     roc_auc = []
#     f1s = []

#     # possible architectures for this n_layers
#     combinations_lay = list(product(*[archs['n_units']]*n_lay))
#     # drop combinations without at least n_lay * 300 neurons
#     combinations_lay = [comb for comb in combinations_lay if np.sum(comb)>=300*n_lay]

#     combinations_drop = list(product(*[archs['dropout']]*n_lay))
#     combinations = list(product(combinations_lay,combinations_drop))
   
#     # pick randomly only 20 combinations
#     if len(combinations) > 20: combinations = random.sample(combinations, 20)
    
#     for comb in combinations:

#         layers = []
#         name_layers = ''
#         name_dropout = ''
#         for unit,drop in zip(comb[0],comb[1]):
#             layers.append(tf.keras.layers.Dense(unit, activation=tf.keras.activations.relu))
#             name_layers = name_layers + f'{unit}x'
            
#             # add dropout layer?
#             if drop:
#                 layers.append(tf.keras.layers.Dropout(drop))
            
#             name_dropout = name_dropout + f'{drop :.1f}+'
                    
#         layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
#         name = str(n_lay) + f'__{learning_rate}__' + name_layers[:-1] + '___' + name_dropout[:-1]

#         compile_params = dict(
#             optimizer = tf.keras.optimizers.Adam(learning_rate= float(learning_rate)),
#             loss = tf.keras.losses.BinaryCrossentropy(),
#             metrics=[]   
#         )
        
#         print ('-'*70)
#         print(f'MODEL: {name}')
#         mod = NNModelController(layers= layers, name= 'bcg_run2/' + name, data= data, compile_params= compile_params, epochs= 1000, batch_size= 2000)
#         mod.main(read_data= False, prep_data= False)

#         names.append(name)
#         losses.append(mod.tester.test_loss)
#         pr_auc.append(mod.tester.pr_auc)
#         roc_auc.append(mod.tester.roc_auc)
#         f1s.append(mod.tester.f1)
        
#     df = pd.DataFrame(dict(model = names, loss = losses, pr_auc = pr_auc, roc_auc = roc_auc, f1 = f1s))
#     df.to_csv('bcg_search_summary_2.csv', index= False, mode= 'a')





