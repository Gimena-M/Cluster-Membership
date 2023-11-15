"""
Class to select galaxies that are within a certain redshift range from the nearest BCG.

Arguments for initialization are:
    * name (string): Name of .csv file without extension. 
    * dir (string): Directory containing the files.
    * df_gal (pd.DataFrame or None): DataFrame with galaxies. If None, the table is read from the file. (default: None)

The main() method performs all the tasks. 
"""

import multiprocessing
import pandas as pd

def dz(z):
    return 0.2 + z * 0.25 # function dz computes a z range, as a function of cluster's z


class RedshiftLimits:

    def __init__(self, name: str, dir: str, df_gal: pd.DataFrame|None = None) -> None:
        self.name = name #name of files
        self.dir = dir #directory for reading and saving files
        self.df_gal = df_gal
        print(f'LIMITING REDSHIFT ON FILE {self.name}')

    def main(self, njobs = 1):
        self._read_files()
        self._z_intervals()
        self._new_dataframe(njobs)
        self._save()
        return self.df_gal

    def _read_files(self):
        print('Reading files...')
        if self.df_gal is None: self.df_gal = pd.read_csv(f'{self.dir}clean_tables/{self.name}.csv')
        self.df_cl =pd.read_table(f'{self.dir}Wen+Han/clusters.dat', delim_whitespace=True, usecols=[0,3,4,5,9,11,12], names=['id_cl','ra_cl','dec_cl','phot_z_cl', 'r500_cl','mass_cl','n500_cl'])

        # drop clusters not in the galaxies df. 
        self.df_cl = self.df_cl[(self.df_cl['id_cl'].isin(self.df_gal.id_cl_near)) | (self.df_cl['id_cl'].isin(self.df_gal.id_cl))]

    def _z_intervals(self):
        print('Computing z intervals...', flush= True)
        # add dz as a column of clusters' DataFrame
        self.df_cl = self.df_cl.assign(d_phot_z = dz(self.df_cl['phot_z_cl'].values))

    def _new_dataframe(self, njobs = 1):
        print('Creating new DataFrame...', flush= True)

        # create a new DataFrame with galaxies inside dz of each cluster
        # df_new = pd.DataFrame()
        d_phot_z = self.df_cl.d_phot_z
        id_cl = self.df_cl.id_cl
        phot_z_cl = self.df_cl.phot_z_cl

        if njobs == -1: njobs = None
        with multiprocessing.Pool(njobs) as p:
            dfs = p.map(self._new_dataframe_one_cluster, zip(id_cl,phot_z_cl,d_phot_z))
        df_new = pd.concat(dfs)

        # for id,d_z,z in zip(id_cl, d_phot_z, phot_z_cl):
        #     g = self.df_gal[self.df_gal.id_cl_near == id]
        #     g = g[abs(g.phot_z - z) <= d_z]
        #     df_new = pd.concat([df_new, g])

        df_new = df_new.drop_duplicates(subset=['object_id'])
        
        print(f'\t Members before: {self.df_gal[self.df_gal.member == 1].shape[0]/self.df_gal.shape[0] * 100 :.2f}%   ({self.df_gal[self.df_gal.member == 1].shape[0]})', flush= True)
        self.df_gal = df_new
        print(f'\t Members after: {self.df_gal[self.df_gal.member == 1].shape[0]/self.df_gal.shape[0] * 100 :.2f}%  ({self.df_gal[self.df_gal.member == 1].shape[0]})', flush=True)

    def _new_dataframe_one_cluster(self, tup: tuple):
        id,z,d_z = tup
        g = self.df_gal[self.df_gal.id_cl_near == id]
        g = g[abs(g.phot_z - z) <= d_z]
        return g

    def _save(self):
        print('Saving...')
        self.df_gal.drop_duplicates(subset = ['ra','dec'])
        self.df_gal.to_csv(f'{self.dir}clean_tables/z_filtered_{self.name}.csv', index= False)
