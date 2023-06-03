"""
This script removes galaxies from a table if their redshift values differ significantly from that of the nearest BCG.
Galaxies are removed if the difference between their redshift and the redshift of the nearest BCG is larger than dz = 0.1 + z * 0.15

From command line: python cut-z.py HSC-unWISE-W01.csv
Argument: csv table
"""

import pandas as pd
import sys

file_df = sys.argv[1]


print('Reading files...', flush= True)

# import galaxies and clusters tables
df = pd.read_csv(file_df)
df = df.drop_duplicates()
df_cl = pd.read_table('../../clusters.dat', delim_whitespace=True, usecols=[0,3,4,5,9,11,12], names=['id_cl','ra_cl','dec_cl','phot_z_cl', 'r500_cl','mass_cl','n500_cl'])

# drop clusters not in the galaxies df. 
df_cl = df_cl[(df_cl['id_cl'].isin(df.id_cl_near)) | (df_cl['id_cl'].isin(df.id_cl))]

# evaluate the column "id_cl_near" (it's a list stored as a string, has the ids of the nearest clusters)
# it's a list and not a single value because some galaxies were near 2 or more clusters
df['id_cl_near'] = [eval(s) for s in df['id_cl_near']]




print('Computing z intervals...', flush= True)

# function dz computes a z range, as a function of cluster's z
def dz(z):
    return 0.1 + z * 0.15

# add dz as a column of clusters' DataFrame
df_cl['d_phot_z'] = dz(df_cl['phot_z_cl'])




print('Creating new DataFrame... (this takes forever)', flush= True)

# create a new DataFrame with galaxies inside dz of each cluster
df_new = pd.DataFrame()
for _,c in df_cl.iterrows():
    # g = df[[any([c.id_cl== ids for ids in gal.id_cl_near]) for _,gal in df.iterrows()]]
    g = df[df['id_cl_near'].apply(lambda x: c.id_cl in x)]
    g = g[abs(g.phot_z - c.phot_z_cl) <= c.d_phot_z]
    df_new = pd.concat([df_new, g])

df_new = df_new.drop_duplicates(subset=['object_id','id_cl'])


print(f'\t Members before: {df[df.member == 1].shape[0]/df.shape[0] * 100 :.2f}%   ({df[df.member == 1].shape[0]})', flush= True)
print(f'\t Members after: {df_new[df_new.member == 1].shape[0]/df_new.shape[0] * 100 :.2f}%  ({df_new[df_new.member == 1].shape[0]})', flush=True)



print('Saving...', flush=True)
df_new.to_csv('z-filtered_'+file_df)