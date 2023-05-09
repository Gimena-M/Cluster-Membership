"""
Remove outliers, NaN and inf from a table.

From command line: python clean.py HSC-unWISE-W01.csv features.txt
Arguments:
    csv table
    text file with a list of features to be checked (one feature in each line)

Features not found in the CSV file are ignored.
Rows with features outside certain limits are removed. Limits are in limits.csv
"""

import pandas as pd
import numpy as np
import csv
import sys

file_df = sys.argv[1]
file_features = sys.argv[2]



print ('Reading files...',flush= True)
#read galaxies
df = pd.read_csv(file_df)
df = df.drop(columns = [f for f in df.columns if ('isnull' in f)])
#read features list
features = []
with open(file_features, mode = 'r') as file:
    features = file.read().splitlines()
    
    
    
print('Removing NaN...', flush= True)
#replace inf for NaN
df = df.replace([np.inf, -np.inf], np.nan)     
df['W2'] = df['W2'].replace(99., np.nan)
df['W1'] = df['W1'].replace(99., np.nan)
df['W2_err'] = df['W2_err'].replace(99., np.nan)
df['W1_err'] = df['W1_err'].replace(99., np.nan)
# check for NaN
for f in features.copy():
    try:
        per= 100*len(np.where(df[f].isnull())[0])/df.shape[0]
        print ('\t{}: {:.4f}% NaN'.format(f, per))  
    except:
        print(f'Feature not found: {f}')
        features.remove(f)
# remove all rows with NaN in 
size0 = df.shape[0]
df = df.dropna(subset = features, axis = 'index')
print('\t {} rows dropped of {}'.format(size0 - df.shape[0], size0))




print('Removing outliers...', flush=True)
# read limits 
limits = dict()
with open('limits.csv', mode = 'r') as file:
    reader = csv.reader(file, delimiter=';')
    next(reader, None) #skip header
    for i in reader:
        limits[i[0]] = (float(i[1]),float(i[2]))     
# drop rows outside of limits
for f in features:
    if f in limits.keys():
        size0 = df.shape[0]
        df = df[ 
            (df[f] >= limits[f][0]) &
            (df[f] <= limits[f][1]) 
        ]
        print('\t{}: {} rows dropped of {}'.format(f, size0 - df.shape[0], size0))
    else:
        print('\t{} not found'.format(f))
        
        
        
print('Saving....', flush=True)
df.to_csv('clean-'+file_df, index = False)