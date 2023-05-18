"""
Match the results from the SQL query with the Wen&Han table (to add the ids to the original table)
From command line : python matching.py HSC-unWISE-W01 
Arguments: name of the .dat file from Wen&Han, and of the .fits file with the query results (without extension)
"""

import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy import coordinates
from astropy import units as u
import grispy
from astropy.table import Table
import sys

file = sys.argv[1]

# read data for clusters and members
df_cl = pd.read_table('0-Wen&Han/clusters.dat', delim_whitespace=True, usecols=[0,3,4,5,9,11,12], names=['id_cl','ra_cl','dec_cl','phot_z_cl', 'r500_cl','mass_cl','n500_cl'])
df_mem = pd.read_table('0-Wen&Han/members.dat', delim_whitespace=True, usecols=[0,1,2,3,18,19], names=['id_cl','ra','dec', 'phot_z', 'log_mass', 'dist'])

# convert r500 and dist in Mpc to degree.
df_cl = df_cl.assign(**{'r500_cl_deg' : df_cl.r500_cl * 1000./(FlatLambdaCDM(H0=70, Om0=0.3).kpc_proper_per_arcmin(df_cl.phot_z_cl.values).value * 60)})
df_mem = df_mem.assign(**{'dist_deg' : df_mem.dist * 1000./(FlatLambdaCDM(H0=70, Om0=0.3).kpc_proper_per_arcmin(df_mem.phot_z.values).value * 60)})

# check if any galaxies are outside r500. if there are, take a larger radius than r500
df_mem = pd.merge(left = df_mem, right=df_cl, on='id_cl')
f = 1.55
while  (df_mem[df_mem.r500_cl_deg * f < df_mem.dist_deg].shape[0] != 0):
      f += 0.01
print("Checking for galaxies inside r_500 * {}".format(f), flush=True)






# read galaxies
df_gal = pd.read_table('0-Wen&Han/' + file + '.dat', delim_whitespace=True, usecols=[0,1,12,13,14,15,16,17,18], names=['ra','dec', 'W1', 'W1_err', 'W2', 'W2_err', 'phot_z', 'phot_z_err', 'log_st_mass'])

# restrict clusters coordinates to those of the galaxies...
df_cl = df_cl[
    ((df_cl.ra_cl >= min(df_gal.ra)) & (df_cl.ra_cl <= max(df_gal.ra))) &
    ((df_cl.dec_cl >= min(df_gal.dec)) & (df_cl.dec_cl <= max(df_gal.dec)))
]
print('Number of clusters: {}'.format(df_cl.shape[0]), flush=True)


if ((max(df_gal.ra) - 360 < 0.001) & (min(df_gal.ra) < 0.001)):
    df_gal['ra2'] = [ra if ra>180 else ra + 360 for ra in df_gal.ra]
    df_gal['dec2'] = [dec if dec>180 else dec + 360 for dec in df_gal.dec]
    df_cl['ra2'] = [ra if ra>180 else ra + 360 for ra in df_cl.ra_cl]
    df_cl['dec2'] = [dec if dec>180 else dec + 360 for dec in df_cl.dec_cl]
    df_mem['ra2'] = [ra if ra>180 else ra + 360 for ra in df_mem.ra]
    df_mem['dec2'] = [dec if dec>180 else dec + 360 for dec in df_mem.dec]
    
    df_cl = df_cl[
        ((df_cl.ra2 >= min(df_gal.ra2)) & (df_cl.ra2 <= max(df_gal.ra2))) &
        ((df_cl.dec2 >= min(df_gal.dec2)) & (df_cl.dec2 <= max(df_gal.dec2)))
    ]
    df_mem = df_mem[
        ((df_mem.ra2 >= min(df_gal.ra2)) & (df_mem.ra2 <= max(df_gal.ra2))) &
        ((df_mem.dec2 >= min(df_gal.dec2)) & (df_mem.dec2 <= max(df_gal.dec2)))
    ]

    df_gal = df_gal.drop(columns = ['ra2', 'dec2'])
    df_mem = df_mem.drop(columns = ['ra2', 'dec2'])
    df_cl = df_cl.drop(columns = ['ra2', 'dec2'])

else:
    # restrict clusters coordinates to those of the galaxies...
    # restrict members coordinates to those of the galaxies...
    df_cl = df_cl[
        ((df_cl.ra_cl >= min(df_gal.ra)) & (df_cl.ra_cl <= max(df_gal.ra))) &
        ((df_cl.dec_cl >= min(df_gal.dec)) & (df_cl.dec_cl <= max(df_gal.dec)))
    ]
    df_mem = df_mem[
        ((df_mem.ra >= min(df_gal.ra)) & (df_mem.ra <= max(df_gal.ra))) &
        ((df_mem.dec >= min(df_gal.dec)) & (df_mem.dec <= max(df_gal.dec)))
    ]





# find galaxies around clusters coordinates (neighbors inside  r500 * f)
grid = grispy.GriSPy(data=df_gal[['ra','dec']].values, metric = 'haversine', periodic = {0: (0,360), 1:None})
index = grid.bubble_neighbors(centres = df_cl[['ra_cl','dec_cl']].values, distance_upper_bound=df_cl.r500_cl_deg.values * f)[1]
del grid

# keep only those galaxies in dataframe. save also the index of the nearest cluster.
df_gal_short = pd.DataFrame()
for i,ind in enumerate(index):
    df_a = df_gal.iloc[ind]
    df_a = df_a.assign(id_cl_near = df_cl['id_cl'].iloc[i])
    df_gal_short = pd.concat([df_gal_short, df_a], axis = 0)
    
df_gal = df_gal_short
df_gal = df_gal.drop_duplicates(subset = ['ra','dec'])
del df_gal_short,df_a    
# df_gal = df_gal.iloc[np.concatenate(index)].drop_duplicates()





# read fits
dat = Table.read('1-sql-ids/'+ file + '.fits', format='fits')
df_sql = dat.to_pandas()
del dat
        
# take the resulting table and match the coordinates to those in df_gal.
gal = coordinates.SkyCoord(ra = df_gal.ra.values * u.deg, dec = df_gal.dec.values * u.deg)
cat = coordinates.SkyCoord(ra = df_sql.ra.values * u.deg, dec = df_sql.dec.values * u.deg)
ids = coordinates.matching.match_coordinates_sky(matchcoord=gal, catalogcoord=cat)[0]

# add object ids
df_gal['object_id'] = df_sql.loc[ids].object_id.values
df_gal['hsc_ra'] = df_sql.loc[ids].ra.values
df_gal['hsc_dec'] = df_sql.loc[ids].dec.values

# save to a csv 
df_gal.to_csv('2-matched-ids/' + file + '.csv', index = False)




#same for members
mem = coordinates.SkyCoord(ra = df_mem.ra.values * u.deg, dec = df_mem.dec.values * u.deg)
ids = coordinates.matching.match_coordinates_sky(matchcoord=mem, catalogcoord=cat)[0]

df_mem['object_id'] = df_sql.loc[ids].object_id.values
df_mem['hsc_ra'] = df_sql.loc[ids].ra.values
df_mem['hsc_dec'] = df_sql.loc[ids].dec.values

df_mem.to_csv('2-matched-ids/members-' + file + '.csv', index = False)
