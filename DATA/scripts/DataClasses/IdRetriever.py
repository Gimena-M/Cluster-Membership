"""
Class to select a set of galaxies from the Wen & Han (2021) catalogues, write and run an SQL query selecting the ids from the HSC-SSP catalogue, and match the results to an existing table.

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

class IdRetriever:

    def __init__(self, name: str, dir: str, user: str) -> None:
        self.name = name #name of files
        self.dir = dir #directory for reading and saving files
        self.user = user
        
        print(f'GETTING IDS FOR {self.name}')

    def main(self, write_query: bool = True, run_query: bool = True):
        self._read_dats()
        self._merge_members_and_clusters()
        self._select_galaxies()
        self._save()
        if write_query: self._write_query_for_ids()
        if run_query: self._run_query_for_ids()
        self._match_ids()
        self._merge_members_and_galaxies()
        self._remove_duplicates()
        self._save()
        return self.df_gal

    def _read_dats(self):
        print('Reading .dat files...', flush= True)

        self.df_cl =pd.read_table(f'{self.dir}Wen+Han/clusters.dat', delim_whitespace=True, usecols=[0,3,4,5,9,11,12], names=['id_cl','ra_cl','dec_cl','phot_z_cl', 'r500_cl','mass_cl','n500_cl'])

        self.df_mem = pd.read_table(f'{self.dir}Wen+Han/members.dat', delim_whitespace=True, usecols=[0,1,2,3,18,19], names=['id_cl','ra','dec', 'phot_z', 'log_mass', 'dist'])

        self.df_gal = pd.read_table(f'{self.dir}Wen+Han/{self.name}.dat', delim_whitespace=True, usecols=[0,1,12,13,14,15,16,17,18], names=['ra','dec', 'W1', 'W1_err', 'W2', 'W2_err', 'phot_z', 'phot_z_err', 'log_st_mass'])

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

    def _merge_members_and_clusters(self):
        print('Joining tables...', flush= True)
         # convert r500 and dist in Mpc to degree.
        self.df_cl = self.df_cl.assign(**{'r500_cl_deg' : self.df_cl.r500_cl * 1000./(cosmo.kpc_proper_per_arcmin(self.df_cl.phot_z_cl.values).value * 60)})

        # add angular distance between member and BCG to members DataFrame.
        # don't use the "dist" with "phot_z" columns (as i did with df_cl), the resulting angular distances are not good.
        ids_bcg = self.df_cl.id_cl.values    # get coords and id of bcgs 
        ras_bcg = self.df_cl.ra_cl.values
        decs_bcg = self.df_cl.dec_cl.values
        df_mem_aux = pd.DataFrame()
        for id_b,ra_b,dec_b in zip(ids_bcg, ras_bcg, decs_bcg):
            mem: pd.DataFrame = self.df_mem[self.df_mem ['id_cl'] == id_b]
            ang_dist_bcg = [haversine(ra,dec,ra_b,dec_b) for ra,dec in zip(mem.ra, mem.dec)]
            mem = mem.assign(ang_dist_bcg= ang_dist_bcg)
            df_mem_aux = pd.concat([df_mem_aux, mem])
        self.df_mem = df_mem_aux

        # check if any galaxies are outside r500. if there are, take a larger radius than r500. If possible, take a smaller one to reduce number of non-members.
        f_r500 = np.full(self.df_cl.shape[0], 0.5)
        r500_cl_deg = self.df_cl.r500_cl_deg.values
        for i,(f500,r500,id) in enumerate(zip(f_r500,r500_cl_deg,ids_bcg)):
            mem = self.df_mem[self.df_mem['id_cl'] == id]
            f = f500
            while (mem[mem.ang_dist_bcg >= r500 * f].shape[0] != 0):
                f += 0.1
            f_r500[i] = round(f, 1)
        self.df_cl = self.df_cl.assign(f_r500= f_r500)
        self.df_mem = pd.merge(left = self.df_mem, right=self.df_cl, on='id_cl', how= 'left')

        print('\t Number of clusters for {}: {}'.format(self.name, self.df_cl.shape[0]), flush=True)

    def _select_galaxies(self):
        
        print('Selecting galaxies...', flush= True)
        
        import grispy
        # find galaxies around clusters coordinates (neighbors inside  r500 * f)
        grid = grispy.GriSPy(data=self.df_gal[['ra','dec']].values, metric = 'haversine', periodic = {0: (0,360), 1:None})
        dists, index = grid.bubble_neighbors(centres = self.df_cl[['ra_cl','dec_cl']].values, distance_upper_bound=self.df_cl.r500_cl_deg.values * self.df_cl.f_r500.values)

        # keep only those galaxies in dataframe. save also the index of the nearest cluster.
        df_gal_short = pd.DataFrame()
        ids_bcg = self.df_cl.id_cl.values 
        for ind,id,dist in zip(index,ids_bcg,dists):
            df_a = self.df_gal.iloc[ind]
            df_a = df_a.assign(id_cl_near = id)
            df_a = df_a.assign(ang_dist_cl_near = dist)
            df_gal_short = pd.concat([df_gal_short, df_a], axis = 0)
        self.df_gal = df_gal_short
        
        # from sklearn.neighbors import NearestNeighbors
        # nn = NearestNeighbors(metric= 'haversine', n_jobs= njobs, algorithm= 'ball_tree')
        # nn.fit(np.deg2rad(self.df_gal[['dec', 'ra']].values))

        # def rn(tup):
        #     return nn.radius_neighbors([tup[0]], tup[1])

        # a = map(rn, zip(np.deg2rad(self.df_cl[['dec_cl', 'ra_cl']].values), self.df_cl.r500_cl_deg.values * self.df_cl.f_r500.values))
        # a = np.array(list(a))
        # dists = a[:,0,0]
        # index = a[:,1,0]      

    def _write_query_for_ids(self):
        print('Writing SQL query for ids', flush= True)

        # write an sql query for a conesearch (ra,dec=degree, rad=arcsec)
        query_line = ("(SELECT "
                "object_id, ra, dec "
                "FROM pdr2_wide.forced "
                "WHERE isprimary='True' AND i_pixelflags_saturatedcenter='False' AND conesearch(coord, {}, {}, {}))")

        query = ''
        for i,(ra,dec,rad,f) in enumerate(zip(self.df_cl.ra_cl.values, self.df_cl.dec_cl.values, self.df_cl.r500_cl_deg.values, self.df_cl.f_r500.values)):
                query = query + query_line.format(ra,dec,rad * f * 3600 + 1) + ' UNION '
        query = query [:-7] #remove final ' UNION '

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

        #same for members
        mem = coordinates.SkyCoord(ra = self.df_mem.ra.values * u.deg, dec = self.df_mem.dec.values * u.deg)
        ids = coordinates.matching.match_coordinates_sky(matchcoord=mem, catalogcoord=cat)[0]
        self.df_mem = self.df_mem.assign(
            object_id = df_sql.loc[ids].object_id.values,
            hsc_ra = df_sql.loc[ids].ra.values,
            hsc_dec = df_sql.loc[ids].dec.values
        )

    def _merge_members_and_galaxies(self):
        print('Joining tables again...', flush= True)
        # join members and clusters
        a = pd.merge(left = self.df_gal, right = self.df_mem, on = 'object_id', how = 'outer')
        if (a[((a.ra_x != a.ra_y) | (a.dec_x != a.dec_y)) & a.ra_y].shape[0] != 0):
            print ('\t There are bad matches...')
        a = a.drop(columns = ['ra_y','dec_y',  'phot_z_y'])
        a = a.rename(columns = {'ra_x' : 'ra', 'dec_x' : 'dec', 'phot_z_x': 'phot_z'})
        a = a.assign(member = [0 if i else 1 for i in a.id_cl.isnull()])
        self.df_gal = a

    def _remove_duplicates(self):
        print('Removing duplicates...', flush=True)
        # some galaxies are near more than one cluster. Those rows will be duplicate. Keep the one with the lower distance to cluster
        dups =  self.df_gal.duplicated(subset=['ra', 'dec'], keep=False).values # array with True for all duplicates
        ras = self.df_gal.ra.values
        decs = self.df_gal.dec.values
        ang_dist_cl_near = self.df_gal.ang_dist_cl_near.values
        id_cl = self.df_gal.id_cl.values
        id_cl_near = self.df_gal.id_cl_near.values
        for i,(dup,ra,dec,dist_cl,id_c,id_c_n) in enumerate(zip(dups,ras,decs,ang_dist_cl_near,id_cl,id_cl_near)):
            if dup:
                if id_c:
                    if id_c == id_c_n:
                        dups[i]=False
                else:
                    min_dist = np.min(self.df_gal[(self.df_gal.ra == ra) & (self.df_gal.dec == dec)]['ang_dist_cl_near'].values)
                    if dist_cl == min_dist:
                        dups[i] = False # choose the row with the cluster with lower distance, and set dups to False
        self.df_gal = self.df_gal[~dups] #keep all non duplicates

        # This is the previous approach to duplicates
        # Store near clusters' ids as a list, and drop duplicates.
        # dups =  self.df_gal.duplicated(subset=['ra','dec'], keep=False)
        # id_cl_near = [np.unique([id for id in self.df_gal[(self.df_gal.ra == gal.ra) & (self.df_gal.dec == gal.dec) ]['id_cl_near']]) 
        #         if d else [int(gal['id_cl_nearest'])] 
        #         for d,(_,gal) in zip(dups,self.df_gal.iterrows()) ]
        # self.df_gal = self.df_gal.assign(id_cl_near = id_cl_near)

        #convert that list to a string, so that it can be saved in a CSV file. When using it, use eval().
        # self.df_gal['id_cl_near'] = '[' + self.df_gal['id_cl_near'].apply(lambda x: ','.join(map(str, x))) + ']'
        self.df_gal = self.df_gal.drop_duplicates(subset = ['ra','dec'])


    def _save(self):
        print('Saving...', flush=True)
        self.df_gal.to_csv(f'{self.dir}unclean_tables/{self.name}.csv', index=False)

    def _save(self):
        print('Saving...', flush=True)
        self.df_gal.to_csv(f'{self.dir}unclean_tables/{self.name}.csv', index=False)


    