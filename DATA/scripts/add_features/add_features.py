"""
Add the following features to a table:  
    * colors g-r, r-i, i-z, z-y
    * colors W1-g, W1-i
    * photometric redshift of the nearest BCG

From command line: python add_features.py  HSC-unWISE-W01.csv
Argument: csv table with galaxies.
"""

import pandas as pd
import numpy as np
import sys

df_file = sys.argv[1]
df = pd.read_csv(df_file)


# add colots
print('Adding colors...')
df = df.assign(gr=df['g_cmodel_mag'] - df['r_cmodel_mag'])
df = df.assign(ri=df['r_cmodel_mag'] - df['i_cmodel_mag'])
df = df.assign(iz=df['i_cmodel_mag'] - df['z_cmodel_mag'])
df = df.assign(zy=df['z_cmodel_mag'] - df['y_cmodel_mag'])
df = df.assign(W1g= df['W1'] - df['g_cmodel_mag'])
df = df.assign(W1i= df['W1'] - df['i_cmodel_mag'])






# add z of the first cluster in the list of id_cl_near, or the cluster that a member galaxy belongs to
print("Adding nearest BCG's redshift...")

# evaluate the nearest clusters' ids columns
df['id_cl_near_eval'] = [np.array(eval(s)) for s in df.id_cl_near]

# working with a column of lists is too slow. 
# explode transforms each element in the column of lists into a separate row.
df = df.explode('id_cl_near_eval')
df.reset_index(drop=True, inplace=True)

df_cl = pd.read_table('../../clusters.dat', delim_whitespace=True, usecols=[0,3,4,5,9,11,12], 
                      names=['id_cl','ra_cl','dec_cl','phot_z_cl', 'r500_cl','mass_cl','n500_cl'])
df['z_cl_near'] = -99

col_i = df.columns.get_loc('z_cl_near') # index of this column
for i,row in df.iterrows():
    if row.member == 0:
        # df.iloc[i,col_i] = df_cl[df_cl.id_cl == row.id_cl_near_eval[0]]['phot_z_cl'].values[0]
        df.iloc[i,col_i] = df_cl[df_cl.id_cl == row.id_cl_near_eval]['phot_z_cl'].values[0]
    else:
        df.iloc[i,col_i] = df_cl[df_cl.id_cl == row.id_cl]['phot_z_cl'].values[0]






print('Saving...')
df.drop_duplicates()
df = df.drop(columns= ['id_cl_near_eval'])
df.to_csv(df_file, index= False)