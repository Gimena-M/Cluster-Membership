"""
Class to add colors and the nearest BCG's redshift to a table.

Arguments for initialization are:
    * name (string): Name of .csv file without extension. 
    * dir (string): Directory containing the files.
    * df_gal (pd.DataFrame or None): DataFrame with galaxies. If None, the table is read from the file. (default: None)

The main() method performs all the tasks. 
"""

import multiprocessing
import pandas as pd

class NewFeatures:

    def __init__(self, name: str, dir: str, df_gal: pd.DataFrame|None = None) -> None:
        self.name = name #name of files
        self.dir = dir #directory for reading and saving files
        self.df_gal = df_gal
        print(f'NEW FEATURES FOR {self.name}')

    def main(self, njobs = 1):
        if self.df_gal is None: self._read_csv()
        self._colors()
        self._near_BCG_z(njobs)
        self._save()
        return self.df_gal

    def _read_csv(self):
        print('Reading .csv...')
        self.df_gal = pd.read_csv(f'{self.dir}clean_tables/{self.name}.csv')

    def _colors(self):
        print('Adding colors...')
        self.df_gal = self.df_gal.assign(
            gr=self.df_gal['g_cmodel_mag'] - self.df_gal['r_cmodel_mag'],
            ri=self.df_gal['r_cmodel_mag'] - self.df_gal['i_cmodel_mag'],
            iz=self.df_gal['i_cmodel_mag'] - self.df_gal['z_cmodel_mag'],
            zy=self.df_gal['z_cmodel_mag'] - self.df_gal['y_cmodel_mag'],
            W1g= self.df_gal['W1'] - self.df_gal['g_cmodel_mag'],
            W1r= self.df_gal['W1'] - self.df_gal['r_cmodel_mag'],
            W1i= self.df_gal['W1'] - self.df_gal['i_cmodel_mag'],
            W1z= self.df_gal['W1'] - self.df_gal['z_cmodel_mag'],
            W1y= self.df_gal['W1'] - self.df_gal['y_cmodel_mag']
        )

    def _near_BCG_z(self, njobs = 1):
        # add z of the cluster of id_cl_near
        print("Adding nearest BCG's redshift...")

        self.df_cl = pd.read_table(f'{self.dir}Wen+Han/clusters.dat', sep='\s+', usecols=[0,3,4,5,9,11,12], names=['id_cl','ra_cl','dec_cl','phot_z_cl', 'r500_cl','mass_cl','n500_cl'])

        # z_cl_near = np.full(self.df_gal.shape[0],-99)
        member = self.df_gal.member.values
        id_cl = self.df_gal.id_cl.values
        id_cl_near = self.df_gal.id_cl_near.values
        # for i,(mem,id_true,id_near) in enumerate(zip(member,id_cl,id_cl_near)):
        #     if mem == 0:
        #         z_cl_near[i] = self.df_cl[self.df_cl.id_cl == id_near]['phot_z_cl'].values[0]
        #     else:
        #         z_cl_near[i] = self.df_cl[self.df_cl.id_cl == id_true]['phot_z_cl'].values[0]
        
        if njobs == -1: njobs = None
        with multiprocessing.Pool(njobs) as p:
            z_cl_near = p.map(self._find_near_BCG_z, zip(member,id_cl_near,id_cl))
        self.df_gal = self.df_gal.assign(z_cl_near = z_cl_near)
        

    def _find_near_BCG_z(self, tup: tuple):
        mem, id_near, id_true = tup
        if mem == 0:
            return self.df_cl[self.df_cl.id_cl == id_near]['phot_z_cl'].values[0]
        else:
            return self.df_cl[self.df_cl.id_cl == id_true]['phot_z_cl'].values[0]

    def _save(self):
        print('Saving...')
        self.df_gal.drop_duplicates(subset = ['ra','dec'])
        self.df_gal.to_csv(f'{self.dir}clean_tables/{self.name}.csv', index= False)
