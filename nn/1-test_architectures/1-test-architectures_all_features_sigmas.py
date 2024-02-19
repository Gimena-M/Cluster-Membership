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

data = DataHandler(validation_sample= True, features_txt= 'all_features_sigmas.txt', balance= 'weights')
data.main()

layers = [
    tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ]

compile_params = dict(
    optimizer = tf.keras.optimizers.Adam(learning_rate= 2e-5),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics=[]   
)

name = 'all_features_sigmas'

mod = NNModelController(layers= layers, name= name, data= data, compile_params= compile_params, epochs= 1000, batch_size= 2000)
mod.main(read_data= False, prep_data= False)


#-----------------------------
# Architecture searches

# Load and prepare data data 
# data = DataHandler(validation_sample= True, features_txt= 'all_features_sigmas.txt', fields_list=['W06'], balance= 'weights')
# data.main()

# 1st run
# Architecture parameters:
# archs = dict(
#     n_layers = [2,3,4], 
#     n_units = [32,64,128,256,512],
#     dropout = [0, 0.2, 0.5],
# )

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
   
#     # pick randomly only 40 combinations
#     if len(combinations) > 40: combinations = random.sample(combinations, 40)
    
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
#         name = str(n_lay) + '__5e-5__' + name_layers[:-1] + '___' + name_dropout[:-1]

#         compile_params = dict(
#             optimizer = tf.keras.optimizers.Adam(learning_rate= 5e-5),
#             loss = tf.keras.losses.BinaryCrossentropy(),
#             metrics=[]   
#         )
        
#         print ('-'*70)
#         print(f'MODEL: {name}')
#         mod = NNModelController(layers= layers, name= 'sigmas_run1/' + name, data= data, compile_params= compile_params, epochs= 1000, batch_size= 2000)
#         mod.main(read_data= False, prep_data= False)

#         names.append(name)
#         losses.append(mod.tester.test_loss)
#         pr_auc.append(mod.tester.pr_auc)
#         roc_auc.append(mod.tester.roc_auc)
#         f1s.append(mod.tester.f1)



        
# df = pd.DataFrame(dict(model = names, loss = losses, pr_auc = pr_auc, roc_auc = roc_auc, f1 = f1s))
# df.to_csv('sigmas_search_summary_1.csv', index= False, mode= 'a')



# 2nd run
# 2 or 3 layers with 512 neuron first layer was better, and more than 100 neurons in the rest
# including dropout was better 
# archs = dict(
#     n_layers = [2,3], 
#     n_units = [32,64,128,256,512],
#     dropout = [0, 0.2, 0.5],
# )

# names = []
# losses = []
# pr_auc = []
# roc_auc = []
# f1s = []

# for n_lay in archs['n_layers']:

#     # possible architectures for this n_layers
#     combinations_lay = list(product(*[archs['n_units']]*n_lay))
#     combinations_drop = list(product(*[archs['dropout']]*n_lay))

#     # keep only combinations first layer of 512 neurons
#     combinations_lay = [comb for comb in combinations_lay if comb[0] == 512]

#     # keep only combinations with a combined number of neurons of at least 512+128
#     combinations_lay = [comb for comb in combinations_lay if np.sum(comb) >= 512+128]

#     # keep only combinations at most one layer without dropout
#     combinations_drop = [comb for comb in combinations_drop if np.where(np.array(comb)==0)[0].shape[0] <= 1]

#     combinations = list(product(combinations_lay,combinations_drop))
#     # pick randomly only 60 combinations
#     if len(combinations) > 60: combinations = random.sample(combinations, 60)
    
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
#         name = str(n_lay) + '__5e-5__' + name_layers[:-1] + '___' + name_dropout[:-1]

#         compile_params = dict(
#             optimizer = tf.keras.optimizers.Adam(learning_rate= 5e-5),
#             loss = tf.keras.losses.BinaryCrossentropy(),
#             metrics=[]   
#         )
        
#         print ('-'*70)
#         print(f'MODEL: {name}')
#         mod = NNModelController(layers= layers, name= 'sigmas_run2/' + name, data= data, compile_params= compile_params, epochs= 1000, batch_size= 2000)
#         mod.main(read_data= False, prep_data= False)

#         names.append(name)
#         losses.append(mod.tester.test_loss)
#         pr_auc.append(mod.tester.pr_auc)
#         roc_auc.append(mod.tester.roc_auc)
#         f1s.append(mod.tester.f1)
        
# df = pd.DataFrame(dict(model = names, loss = losses, pr_auc = pr_auc, roc_auc = roc_auc, f1 = f1s))
# df.to_csv('sigmas_search_summary_2.csv', index= False, mode= 'a')




