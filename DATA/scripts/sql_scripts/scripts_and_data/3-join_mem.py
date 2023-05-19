"""
Merge galaxies and members tables.
From command line : python join-mem.py HSC-unWISE-W01 
Arguments: name of the csv file with galaxies (without extension)
The name of the members file should be members-HSC-unWISE-W01.csv
"""

import pandas as pd
import sys

file = sys.argv[1]

# read galaxy files
df_gal = pd.read_csv(f'2-matched-ids/{file}.csv')
df_mem = pd.read_csv(f'2-matched-ids/members-{file}.csv')
# df_cl = pd.read_table('data/clusters.dat', delim_whitespace=True, usecols=[0,3,4,5,9,11,12], names=['id_cl','ra_cl','dec_cl','phot_z_cl','r500_cl','mass_cl','n500_cl'])

# join members and clusters
a = pd.merge(left = df_gal, right = df_mem, on = 'object_id', how = 'outer')
if (a[((a.ra_x != a.ra_y) | (a.dec_x != a.dec_y)) & a.ra_y].shape[0] != 0):
    print ('There are bad matches...')
a = a.drop(columns = ['ra_y','dec_y',  'hsc_ra_y', 'hsc_dec_y', 'phot_z_y'])
a = a.rename(columns = {'ra_x' : 'ra', 'dec_x' : 'dec', 'hsc_ra_x' : 'hsc_ra', 'hsc_dec_x' : 'hsc_dec', 'phot_z_x': 'phot_z'})
a['member'] = [0 if i else 1 for i in a.id_cl.isnull()]

a.to_csv(f'3-joined-mem-gal/{file}.csv', index=False)