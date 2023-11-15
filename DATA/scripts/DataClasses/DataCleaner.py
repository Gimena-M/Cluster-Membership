"""
Class to clean galaxy tables by removing invalid values and outliers.

Arguments for initialization are:
    * name (string): Name of .csv file without extension. 
    * dir (string): Directory containing the files.
    * df_gal (pd.DataFrame or None): DataFrame with galaxies. If None, the table is read from the file. (default: None)

The main() method performs all the data cleansing tasks.
"""

import numpy as np
import pandas as pd
import csv

class DataCleaner:

    def __init__(self, name: str, dir: str, df_gal: pd.DataFrame|None = None) -> None:
        self.name = name #name of files
        self.dir = dir #directory for reading and saving files
        self.df_gal = df_gal
        print(f'CLEANING FILE {self.name}')

    def main(self):
        self._read_files()
        self._remove_nan()
        self._remove_outliers()
        self._save()
        return self.df_gal

    def _read_files(self):
        print ('Reading files...',flush= True)

        #read galaxies
        if self.df_gal is None: self.df_gal = pd.read_csv(f'{self.dir}unclean_tables/{self.name}.csv')
        self.df_gal = self.df_gal.drop(columns = [f for f in self.df_gal.columns if ('isnull' in f)])

        #read features list
        self.features = []
        with open(f'{self.dir}clean_tables/clean_features.txt', mode = 'r') as file:
            self.features = file.read().splitlines()

        # read limits 
        self.limits = dict()
        with open(f'{self.dir}clean_tables/limits.csv', mode = 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader, None) #skip header
            for i in reader:
                self.limits[i[0]] = (float(i[1]),float(i[2]))

    def _remove_nan(self):
        print('Removing NaN...', flush= True)

        #replace inf for NaN
        self.df_gal = self.df_gal.replace([np.inf, -np.inf], np.nan)
        self.df_gal['W2'] = self.df_gal['W2'].replace(99., np.nan)
        self.df_gal['W1'] = self.df_gal['W1'].replace(99., np.nan)
        self.df_gal['W2_err'] = self.df_gal['W2_err'].replace(99., np.nan)
        self.df_gal['W1_err'] = self.df_gal['W1_err'].replace(99., np.nan)

        # check for NaN
        for f in self.features.copy():
            try:
                per= 100*len(np.where(self.df_gal[f].isnull())[0])/self.df_gal.shape[0]
                print ('\t{}: {:.4f}% NaN'.format(f, per))
            except:
                print(f'Feature not found: {f}')
                self.features.remove(f)
        # remove all rows with NaN in 
        size0 = self.df_gal.shape[0]
        self.df_gal = self.df_gal.dropna(subset = self.features, axis = 'index')
        print('\t {} rows dropped of {}'.format(size0 - self.df_gal.shape[0], size0))

    def _remove_outliers(self):
        print('Removing outliers...', flush=True)

        # drop rows outside of limits
        for f in self.features:
            if f in self.limits.keys():
                size0 = self.df_gal.shape[0]
                self.df_gal = self.df_gal[
                    (self.df_gal[f] >= self.limits[f][0]) &
                    (self.df_gal[f] <= self.limits[f][1])
                ]
                print('\t{}: {} rows dropped of {}'.format(f, size0 - self.df_gal.shape[0], size0))
            else:
                print('\t{} not found'.format(f))

    def _save(self):
        print('Saving...', flush=True)
        self.df_gal = self.df_gal.drop_duplicates(subset= ['ra','dec'])
        self.df_gal.to_csv(f'{self.dir}clean_tables/{self.name}.csv', index= False)
