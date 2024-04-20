"""
Class to add absolute colors to a table.

Arguments for initialization are:
    * name (string): Name of .csv file without extension. 
    * dir (string): Directory containing the files.
    * df_gal (pd.DataFrame or None): DataFrame with galaxies. If None, the table is read from the file. (default: None)

The main() method performs all the tasks. 
"""

import pandas as pd

class AbsoluteColors:

    def __init__(self, name: str, dir: str, df_gal: pd.DataFrame|None = None) -> None:
        self.name = name #name of files
        self.dir = dir #directory for reading and saving files
        self.df_gal = df_gal
        print(f'ABSOLUTE COLORS FOR {self.name}')

    def main(self):
        if self.df_gal is None: self._read_csv()
        self._colors()
        self._save()
        return self.df_gal

    def _read_csv(self):
        print('Reading .csv...')
        self.df_gal = pd.read_csv(f'{self.dir}clean_tables/{self.name}.csv')

    def _colors(self):
        print('Adding colors...')
        self.df_gal = self.df_gal.assign(
            gr_abs=self.df_gal['g_cmodel_mag_abs'] - self.df_gal['r_cmodel_mag_abs'],
            ri_abs=self.df_gal['r_cmodel_mag_abs'] - self.df_gal['i_cmodel_mag_abs'],
            iz_abs=self.df_gal['i_cmodel_mag_abs'] - self.df_gal['z_cmodel_mag_abs'],
            zy_abs=self.df_gal['z_cmodel_mag_abs'] - self.df_gal['y_cmodel_mag_abs'],
        )

    def _save(self):
        print('Saving...')
        self.df_gal.drop_duplicates(subset = ['ra','dec'])
        self.df_gal.to_csv(f'{self.dir}clean_tables/{self.name}.csv', index= False)