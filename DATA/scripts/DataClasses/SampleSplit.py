import numpy as np
import pandas as pd


class SampleSplit:
    
    def __init__(self, name: str, dir: str, df_gal: pd.DataFrame|None = None) -> None:
        self.name = name #name of files
        self.dir = dir #directory for reading and saving files
        self.df_gal = df_gal
        print(f'TRAIN-TEST-VALIDATION SPLIT FOR {self.name}')

    def main(self):
        if self.df_gal is None: self._read_csv()
        self._split()
        self._save()
        return self.df_gal
    
    def _read_csv(self):
        print('Reading .csv...')
        self.df_gal = pd.read_csv(f'{self.dir}clean_tables/{self.name}.csv')
    
    def _split(self):

        from sklearn.model_selection import train_test_split
        training, testing = train_test_split(self.df_gal, test_size = 0.3, stratify = self.df_gal.member, random_state = 42)
        validation, testing = train_test_split(testing, test_size = 0.3, stratify = testing.member, random_state = 42)

        split = np.full(self.df_gal.shape[0], -9)

        for i,s in enumerate(split):
            if i in training.index:
                split[i]= 0
            elif i in testing.index:
                split[i]= 1
            elif i in validation.index:
                split[i]= 2

        self.df_gal = self.df_gal.assign(split = split)

    def _save(self):
        print('Saving...')
        self.df_gal.drop_duplicates(subset = ['ra','dec'])
        self.df_gal.to_csv(f'{self.dir}clean_tables/{self.name}.csv', index= False)