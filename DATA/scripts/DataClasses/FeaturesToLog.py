import numpy as np
import pandas as pd


class FeauturesToLog:
    
    def __init__(self, name: str, dir: str, df_gal: pd.DataFrame|None = None) -> None:
        self.name = name #name of files
        self.dir = dir #directory for reading and saving files
        self.df_gal = df_gal
        print(f'SIGMAS TO LOG(SIGMAS) FOR {self.name}')

    def main(self):
        if self.df_gal is None: self._read_csv()
        self.n_gal = self.df_gal.shape[0]
        self._log_sigmas()
        self._log_shapes()
        self._log_errors()
        self._save()
        return self.df_gal
    
    def _read_csv(self):
        print('Reading .csv...')
        self.df_gal = pd.read_csv(f'{self.dir}clean_tables/{self.name}.csv')

    def _log_sigmas(self):

        print('Changing to log sigmas...', end = '')
        n_neig = [3,5,7,10]
        vel_lims = [1000, 3000, 5000, 10000]
        names = [f'sigma_{n}' for n in n_neig] + [f'sigma_{n}_{v}' for n in n_neig for v in vel_lims]

        for name in names:
            self._log(name)

        print(f' \t{self.n_gal - self.df_gal.shape[0]} rows dropped out of {self.n_gal}')

    def _log_shapes(self):
        print('Changing to log shapes...', end = '')
        filter = ['r', 'i', 'z', 'y']
        names = [f'{f}_sdssshape_shape{ii}' for ii in ['11', '22'] for f in filter] + [f'{f}_sdssshape_shape{ii}sigma' for ii in ['11', '22', '12'] for f in filter]

        for name in names:
            self._log(name)
        print(f' \t{self.n_gal - self.df_gal.shape[0]} rows dropped out of {self.n_gal}')

    def _log_errors(self):
        print('Changing to log errors...', end = '')
        variables = ['cmodel_magsigma', 'kronflux_magsigma']
        filter = ['g', 'r', 'i', 'z', 'y']
        names = [f'{f}_{var}' for f in filter for var in variables] + ['W1_err']

        for name in names:
            self._log(name)
        print(f' \t{self.n_gal - self.df_gal.shape[0]} rows dropped out of {self.n_gal}')

    def _log(self, name):
        no_log = self.df_gal[name].values
        log = np.log10(no_log)
        self.df_gal[name] = log
        self.df_gal.assign(**{f'{name}__no_log' : no_log})
        self.df_gal.dropna(axis = 0, inplace= True, subset= name)

    def _save(self):
        print('Saving...')
        self.df_gal.drop_duplicates(subset = ['ra','dec'])
        self.df_gal.to_csv(f'{self.dir}clean_tables/{self.name}.csv', index= False)