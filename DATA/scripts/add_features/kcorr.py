"""
Add absolute magnitudes to a table (with galactic extintion correction and Kcorrect).

From command line: python kcorr.py HSC-unWISE-W01.csv
Argument: csv table with galaxies.
"""


from kcorrect.kcorrect import Kcorrect
import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import os
import sys

path = os.path.dirname(__file__) +'/'  #of this file
df_file = sys.argv[1]

filter_files = [# list of files with response functions (from https://hsc.mtk.nao.ac.jp/ssp/survey/#filters_and_depths)
    'hsc_g_v2018',
    'hsc_r2_v2018',
    'hsc_i2_v2018',
    'hsc_z_v2018',
    'hsc_y_v2018'
]

# list of par files for each filter, with ABSOLUT PATH
# Kcorrect does have some response functions for subaru, but it's missing the Y filter. Also, the i and r filters were changed in 2016 (see HSC-SSP DR2 note).
responses = [path + 'par_responses/' + fil for fil in filter_files]
kc = Kcorrect(responses = responses, abcorrect= True, redshift_range= [0., 3.], nredshift = 5000, cosmo= FlatLambdaCDM(H0=70, Om0=0.3))

df = pd.read_csv('../../'+ df_file)
magnitudes = ['g_cmodel_mag', 'r_cmodel_mag', 'i_cmodel_mag', 'z_cmodel_mag', 'y_cmodel_mag']
errors = ['g_cmodel_magsigma', 'r_cmodel_magsigma', 'i_cmodel_magsigma', 'z_cmodel_magsigma', 'y_cmodel_magsigma']
absorption = ['a_g', 'a_r', 'a_i', 'a_z', 'a_y']  # absorption

# correct galactic extintion and convert the magnitudes to "maggies"
# maggies are a linear flux density unit defined as 10^{-0.4 m_AB}  where  m_AB is the AB apparent magnitude. 
# That is, 1 maggie is the flux density in Janskys divided by 3631. 

# https://www.sdss3.org/dr8/algorithms/magnitudes.php
# http://wiki.ipb.ac.rs/index.php/Astro_links

maggies = pd.DataFrame()
ivars = pd.DataFrame()
for mag,err,ab in zip(magnitudes, errors, absorption):
    maggies[mag] = [10**(-0.4 * (m - a)) for m,a in zip(df[mag].values, df[ab].values)]     # 10^[-0.4 * (m - a_ext)]
    ivars[mag]=[1./(0.4*np.log(10.)*maggie*e)**2 for e,maggie in zip(df[err].values, maggies[mag].values)]   # 1. / [0.4 * ln(10) * maggie * m_err]**2


coeffs = kc.fit_coeffs(redshift = df['phot_z'].values, maggies = maggies.values, ivar = ivars.values)  
# the docs says that it transforms maggies to the ab system (so i don't have to do that?)

# “NNLS quitting on iteration count.” 
# This indicates that the default number of iterations for scipy.optimize.nnls was not enough. 
# Under these conditions, this code tries a much larger number of iterations. If that still fails, you will receive a traceback.

abs_mags = kc.absmag(redshift = df['phot_z'].values, maggies = maggies.values, ivar = ivars.values, coeffs = coeffs)


# add to df and save
for i,nam in enumerate(magnitudes):
    df[nam + '_abs'] = abs_mags[:,i]

df.to_csv('../../'+ df_file, index= False)