"""
Class to select a set of FIELD galaxies from the Wen & Han (2021) catalogues, write and run an SQL query selecting the ids from the HSC-SSP catalogue, and match the results to an existing table.

Arguments for initialization are:
    * name (string): Name of .csv file without extension. 
    * dir (string): Directory containing the files.
    * user (string): User account for the HSC site. 

The main() method performs all the tasks.
"""

import math
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

def haversine(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(math.radians, [ra1, dec1, ra2, dec2])
    dlon = ra2 - ra1 
    dlat = dec2 - dec1 
    a = math.sin(dlat/2)**2 + math.cos(dec1) * math.cos(dec2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    return math.degrees(c)

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

class IdRetrieverField:

    def __init__(self, name: str, dir: str, user: str) -> None:
        self.name = name #name of files
        if "FIELD" not in name:
            raise Warning('Did you provide the correct name?')
        self.dir = dir #directory for reading and saving files
        self.user = user
        
        print(f'GETTING IDS FOR {self.name}')

    def main(self, select_galaxies: bool = True, write_query: bool = True, run_query: bool = True):
        if select_galaxies:
            self._read_dats()
            self._r500_to_degree() 
            self._remove_cluster_galaxies()
            self._nearest_cluster_id()
            self._save()
        else:
            self.df_gal = pd.read_csv(f'{self.dir}unclean_tables/{self.name}.csv')
        if write_query: self._write_query_for_ids()
        if run_query: self._run_query_for_ids()
        self._match_ids()
        self._save()
        return self.df_gal

    def _read_dats(self):
        print('Reading .dat files...', flush= True)

        self.df_cl =pd.read_table(f'{self.dir}Wen+Han/clusters.dat', sep='\s+', usecols=[0,3,4,5,9,11,12], names=['id_cl','ra_cl','dec_cl','phot_z_cl', 'r500_cl','mass_cl','n500_cl'])

        # self.df_mem = pd.read_table(f'{self.dir}Wen+Han/members.dat', sep='\s+', usecols=[0,1,2,3,18,19], names=['id_cl','ra','dec', 'phot_z', 'log_mass', 'dist'])

        gal_name = self.name.replace('__FIELD','') 
        self.df_gal = pd.read_table(f'{self.dir}Wen+Han/{gal_name}.dat', sep='\s+', usecols=[0,1,12,13,14,15,16,17,18], names=['ra','dec', 'W1', 'W1_err', 'W2', 'W2_err', 'phot_z', 'phot_z_err', 'log_st_mass'])

        # restrict clusters coordinates to those of the galaxies...
        if ((max(self.df_gal.ra) - 360 < 0.001) & (min(self.df_gal.ra) < 0.001)):

            df_cl_aux = pd.DataFrame()
            df_gal_aux = pd.DataFrame()
            df_gal_aux['ra2'] = [ra if ra>180 else ra + 360 for ra in self.df_gal.ra]
            df_gal_aux['dec2'] = [dec if dec>180 else dec + 360 for dec in self.df_gal.dec]
            df_cl_aux['ra2'] = [ra if ra>180 else ra + 360 for ra in self.df_cl.ra_cl]
            df_cl_aux['dec2'] = [dec if dec>180 else dec + 360 for dec in self.df_cl.dec_cl]

            self.df_cl = self.df_cl[
                ((df_cl_aux.ra2 >= min(df_gal_aux.ra2)) & (df_cl_aux.ra2 <= max(df_gal_aux.ra2))) &
                ((df_cl_aux.dec2 >= min(df_gal_aux.dec2)) & (df_cl_aux.dec2 <= max(df_gal_aux.dec2)))
            ]

        else:
            self.df_cl = self.df_cl[
                ((self.df_cl.ra_cl >= min(self.df_gal.ra)) & (self.df_cl.ra_cl <= max(self.df_gal.ra))) &
                ((self.df_cl.dec_cl >= min(self.df_gal.dec)) & (self.df_cl.dec_cl <= max(self.df_gal.dec)))
            ]

    def _r500_to_degree(self):
        print('Joining tables...', flush= True)
         # convert r500 and dist in Mpc to degree.
        self.df_cl = self.df_cl.assign(**{'r500_cl_deg' : self.df_cl.r500_cl * 1000./(cosmo.kpc_proper_per_arcmin(self.df_cl.phot_z_cl.values).value * 60)})

    def _remove_cluster_galaxies(self):
        
        print('Selecting galaxies...', flush= True)
        
        import grispy
        # find galaxies anywhere near clusters coordinates (inside r500 * 6)
        grid = grispy.GriSPy(data=self.df_gal[['ra','dec']].values, metric = 'haversine', periodic = {0: (0,360), 1:None})
        _, index = grid.bubble_neighbors(centres = self.df_cl[['ra_cl','dec_cl']].values, distance_upper_bound=self.df_cl.r500_cl_deg.values * 6.)

        # remove those galaxies from dataframe.
        index = [i for ind in index for i in ind]
        self.df_gal = self.df_gal.drop([self.df_gal.index[i] for i in index])
    
    def _nearest_cluster_id(self):

        print('Finding nearest clusters...')

        # find the nearest cluster 
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors= 1, metric= 'haversine', n_jobs= -1, algorithm= 'ball_tree')
        nn.fit(np.deg2rad(self.df_cl[['dec_cl', 'ra_cl']].values))

        n_all = nn.kneighbors(np.deg2rad(self.df_gal[['dec', 'ra']].values), return_distance= False)
        self.df_gal['id_cl_near'] = [self.df_cl.iloc[i[0]]['id_cl'] for i in n_all]

    def _write_query_for_ids(self):
        print('Writing SQL query for ids', flush= True)

        # write an sql query for a conesearch (ra,dec=degree, rad=arcsec)
        query = ("SELECT "
                "object_id, ra, dec "
                "FROM pdr2_wide.forced "
                f"WHERE isprimary='True' AND i_pixelflags_saturatedcenter='False' AND boxSearch(coord, {min(self.df_gal.ra.values)}, {max(self.df_gal.ra.values)}, {min(self.df_gal.dec.values)}, {max(self.df_gal.dec.values)})")

        with open(f'{self.dir}SQL_queries/{self.name}_ids.sql', mode = 'w') as s:
            s.write(query)

    def _run_query_for_ids(self):
        print('Running SQL query for ids...', flush= True)
        import os
        command = f'python hscReleaseQuery.py --user {self.user} --format fits --delete-job "SQL_queries/{self.name}_ids.sql" > "SQL_queries/{self.name}_ids.fits" '
        os.system(command)

    def _match_ids(self):
        from astropy.table import Table
        from astropy import coordinates
        from astropy import units as u
        print('Matching query results...', flush= True)

        # read fits
        df_sql = Table.read(f'{self.dir}SQL_queries/{self.name}_ids.fits', format='fits')
        df_sql = df_sql.to_pandas()

        # take the resulting table and match the coordinates to those in df_gal.
        gal = coordinates.SkyCoord(ra = self.df_gal.ra.values * u.deg, dec = self.df_gal.dec.values * u.deg)
        cat = coordinates.SkyCoord(ra = df_sql.ra.values * u.deg, dec = df_sql.dec.values * u.deg)
        ids = coordinates.matching.match_coordinates_sky(matchcoord=gal, catalogcoord=cat)[0]
        # add object ids
        self.df_gal = self.df_gal.assign(
            object_id = df_sql.loc[ids].object_id.values,
            hsc_ra = df_sql.loc[ids].ra.values,
            hsc_dec = df_sql.loc[ids].dec.values
        )        

    def _save(self):
        print('Saving...', flush=True)
        self.df_gal = self.df_gal.drop_duplicates(subset = ['ra','dec'])
        self.df_gal = self.df_gal.assign(member = 0) #add column with membership
        self.df_gal.to_csv(f'{self.dir}unclean_tables/{self.name}.csv', index=False)


    