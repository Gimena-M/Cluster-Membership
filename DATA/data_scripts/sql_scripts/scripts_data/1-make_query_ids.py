"""
For each field, write an SQL query for a conesearch around each cluster. 
"""

import pandas as pd
from astropy.cosmology import FlatLambdaCDM


# read data for clusters and members
df_cl = pd.read_table('0-Wen&Han/clusters.dat', delim_whitespace=True, usecols=[0,3,4], names=['id_cl','ra_cl','dec_cl'])
# usecols=[0,3,4,5,9,11,12], names=['id_cl','ra_cl','dec_cl','phot_z_cl', 'r500_cl','mass_cl','n500_cl']
df_mem = pd.read_table('0-Wen&Han/members.dat', delim_whitespace=True, usecols=[0,1,2,19], names=['id_cl','ra','dec', 'dist'])
# usecols=[0,1,2,3,18,19], names=['id_cl','ra','dec', 'phot_z', 'log_mass', 'dist']

list_gal = ['HSC-unWISE-W01.dat','HSC-unWISE-W02.dat','HSC-unWISE-W03.dat','HSC-unWISE-W04.dat','HSC-unWISE-W05.dat','HSC-unWISE-W06.dat','HSC-unWISE-W07.dat']

# convert r500 and dist in Mpc to degree.
df_cl = df_cl.assign(**{'r500_cl_deg' : df_cl.r500_cl * 1000./(FlatLambdaCDM(H0=70, Om0=0.3).kpc_proper_per_arcmin(df_cl.phot_z_cl.values).value * 60)})
df_mem = df_mem.assign(**{'dist_deg' : df_mem.dist * 1000./(FlatLambdaCDM(H0=70, Om0=0.3).kpc_proper_per_arcmin(df_mem.phot_z.values).value * 60)})

# check if any galaxies are outside r500
# if there are, take a larger radius than r500
df_mem = pd.merge(left = df_mem, right=df_cl, on='id_cl')
f = 1
while  (df_mem[df_mem.r500_cl_deg * f < df_mem.dist_deg].shape[0] != 0):
     f += 0.01
print("Looking for galaxies inside r_500 * {}".format(f), flush=True)
del df_mem

for g in list_gal:
    df = pd.read_table('0-Wen&Han/' + g, delim_whitespace=True, usecols=[0,1], names=['ra','dec'])
    # usecols=[0,1,16,17,18], names=['ra','dec','phot_z', 'phot_z_err', 'log_st_mass']
    
    # restrict clusters coordinates to those of the galaxies...
    if ((max(df.ra) - 360 < 0.001) & (min(df.ra) < 0.001)):
        print(g, flush=True)
        df['ra2'] = [ra if ra>180 else ra + 360 for ra in df.ra]
        df['dec2'] = [dec if dec>180 else dec + 360 for dec in df.dec]
        df_cl['ra2'] = [ra if ra>180 else ra + 360 for ra in df_cl.ra_cl]
        df_cl['dec2'] = [dec if dec>180 else dec + 360 for dec in df_cl.dec_cl]
        
        cl = df_cl[
            ((df_cl.ra2 >= min(df.ra2)) & (df_cl.ra2 <= max(df.ra2))) &
            ((df_cl.dec2 >= min(df.dec2)) & (df_cl.dec2 <= max(df.dec2)))
        ]
   
    else:
        cl = df_cl[
            ((df_cl.ra_cl >= min(df.ra)) & (df_cl.ra_cl <= max(df.ra))) &
            ((df_cl.dec_cl >= min(df.dec)) & (df_cl.dec_cl <= max(df.dec)))
        ]
            
    
    print('Number of clusters for {}: {}'.format(g, cl.shape[0]), flush=True)
    # find galaxies around clusters coordinates. 

    # write an sql query for a conesearch (ra,dec=degree, rad=arcsec)
    query_line = ("(SELECT "
             "object_id, ra, dec "
             "FROM pdr2_wide.forced "
             "WHERE isprimary='True' AND i_pixelflags_saturatedcenter='False' AND conesearch(coord, {}, {}, {}))")
    
    query = ''
    for i,(ra,dec,rad) in enumerate(zip(cl.ra_cl.values, cl.dec_cl.values, cl.r500_cl_deg.values * f)):
            query = query + query_line.format(ra,dec,rad * 3600 + 5) + ' UNION '
    query = query [:-7] #remove final ' UNION '
    
    with open('1-sql-ids/{}.sql'.format(g[:-4]), mode = 'x') as s:
        s.write(query)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    