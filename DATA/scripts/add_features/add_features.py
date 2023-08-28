"""
Add the following features to a table:  
    * colors g-r, r-i, i-z, z-y
    * colors W1-g, W1-i
    * photometric redshift of the nearest BCG

From command line: python add_features.py  HSC-unWISE-W01.csv HSC-unWISE-W01.dat
Arguments: 
    csv table with galaxies.
    dat table with all objects (published by Wen & Han)
"""

import pandas as pd
import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors
from astropy.cosmology import FlatLambdaCDM

df_file = sys.argv[1]
dat_file = sys.argv[2]
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




print('Adding sigma_5...')
dat = pd.read_table(dat_file, delim_whitespace=True, usecols=[0,1], names=['ra','dec'])

nbrs = NearestNeighbors(n_neighbors= 6, algorithm='ball_tree', n_jobs = -1, metric= 'haversine') # the first neighbor is the point itself
nbrs.fit(np.deg2rad(dat[['dec', 'ra']].values)) # fit it to the whole sample
n_all = nbrs.kneighbors(np.deg2rad(df[['dec', 'ra']].values), return_distance= True)  # find neighbors only for my sample
ang_dist_5 = np.rad2deg(n_all[0][:,5])  # angular distances to 5th neighbor

# convert to Mpc
lcmd = FlatLambdaCDM(H0=70, Om0=0.3)
mpc_deg = lcmd.kpc_proper_per_arcmin(df['phot_z'].values).value / 1000 * 60  # Mpc/degree
dist_5 = ang_dist_5 * mpc_deg

df['sigma_5'] = 5/ np.pi /dist_5**2



print('Saving...')
df.drop_duplicates()
df = df.drop(columns= ['id_cl_near_eval'])
df.to_csv(df_file, index= False)