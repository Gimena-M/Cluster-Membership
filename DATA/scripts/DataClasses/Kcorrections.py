"""
Class to add K corrections and absolute magnitudes to a table.

Arguments for initialization are:
    * name (string): Name of .csv file without extension. 
    * dir (string): Directory containing the files.
    * df_gal (pd.DataFrame or None): DataFrame with galaxies. If None, the table is read from the file. (default: None)

The main() method performs all the tasks. 
"""

from kcorrect.kcorrect import Kcorrect
import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

class Kcorrections:

    def __init__(self, name: str, dir: str, df_gal: pd.DataFrame|None = None) -> None:
        self.name = name #name of files
        self.dir = dir #directory for reading and saving files
        self.df_gal = df_gal
        print(f'KCORRECTIONS FOR {self.name}')

        self.filter_files = [# list of files with response functions (from https://hsc.mtk.nao.ac.jp/ssp/survey/#filters_and_depths)
            'hsc_g_v2018',
            'hsc_r2_v2018',
            'hsc_i2_v2018',
            'hsc_z_v2018',
            'hsc_y_v2018'
        ]

        self.magnitudes = ['g_cmodel_mag', 'r_cmodel_mag', 'i_cmodel_mag', 'z_cmodel_mag', 'y_cmodel_mag']
        self.errors = ['g_cmodel_magsigma', 'r_cmodel_magsigma', 'i_cmodel_magsigma', 'z_cmodel_magsigma', 'y_cmodel_magsigma']
        self.absorption = ['a_g', 'a_r', 'a_i', 'a_z', 'a_y']  

    def main(self):
        self._read_files()
        self._kcorr()
        self._save()
        return self.df_gal


    def _read_files(self):
        print('Reading files...')
        self.df_gal = pd.read_csv(f'{self.dir}clean_tables/{self.name}.csv')

    def _kcorr(self):

        responses = [self.dir + 'clean_tables/par_responses/' + fil for fil in self.filter_files]
        self.kc = Kcorrect(responses = responses, abcorrect= False, redshift_range= [0., 3.], nredshift = 5000, cosmo= cosmo)

        # correct galactic extintion and convert the magnitudes to "maggies"
        # maggies are a linear flux density unit defined as 10^{-0.4 m_AB}, where m_AB is the AB apparent magnitude. 
        # That is, 1 maggie is the flux density in Janskys divided by 3631. 

        # https://www.sdss3.org/dr8/algorithms/magnitudes.php
        # http://wiki.ipb.ac.rs/index.php/Astro_links

        maggies = pd.DataFrame()
        ivars = pd.DataFrame()
        for mag,err,ab in zip(self.magnitudes, self.errors, self.absorption):
            maggies[mag] = [10**(-0.4 * (m - a)) for m,a in zip(self.df_gal[mag].values, self.df_gal[ab].values)]     # 10^[-0.4 * (m - a_ext)]
            ivars[mag]=[1./(0.4*np.log(10.)*maggie*e)**2 for e,maggie in zip(self.df_gal[err].values, maggies[mag].values)]   # 1. / [0.4 * ln(10) * maggie * m_err]**2

        coeffs = self.kc.fit_coeffs(redshift = self.df_gal['phot_z'].values, maggies = maggies.values, ivar = ivars.values)  

        # “NNLS quitting on iteration count.” 
        # This indicates that the default number of iterations for scipy.optimize.nnls was not enough. 
        # Under these conditions, this code tries a much larger number of iterations. If that still fails, you will receive a traceback.
    
        kcorr = self.kc.kcorrect(redshift = self.df_gal['phot_z'].values, coeffs = coeffs)
        abs_mags = self.kc.absmag(redshift = self.df_gal['phot_z'].values, maggies = maggies.values, ivar = ivars.values, coeffs = coeffs)


        # add to df and save
        for i,nam in enumerate(self.magnitudes):
            self.df_gal[nam + '_abs'] = abs_mags[:,i]
            self.df_gal[nam + 'k_corr'] = kcorr[:,i]

        
    def _save(self):
        print('Saving...')
        self.df_gal.drop_duplicates(subset = ['ra','dec'])
        self.df_gal.to_csv(f'{self.dir}clean_tables/{self.name}.csv', index= False)
