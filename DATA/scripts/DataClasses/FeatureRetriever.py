"""
Class to write and run an SQL query selecting a set of features from the HSC-SSP catalogue, and match the results to an existing table.

Arguments for initialization are:
    * name (string): Name of .csv file without extension. 
    * dir (string): Directory containing the files.
    * df_gal (pd.DataFrame or None): DataFrame with galaxies. If None, the table is read from the file. (default: None)

The main() method performs all the tasks. The selected features are in the columns.txt file.
"""

import pandas as pd

class FeatureRetriever:
    
    def __init__(self, name: str, dir: str, user: str, df_gal: pd.DataFrame|None = None) -> None:
        self.name = name #name of files
        self.dir = dir #directory for reading and saving files
        self.df_gal = df_gal
        self.user = user
        print(f'GETTING FEATURES FOR {self.name}')

    def main(self, write_query: bool = True, run_query: bool = True):
        if self.df_gal is None: self._read_csv()
        if write_query: self._write_query_for_features()
        if run_query: self._run_query_for_features()
        self._match_results()
        return self.df_gal

    def _read_csv(self):
        print('Reading .csv file...', flush= True)
        self.df_gal = pd.read_csv(f'{self.dir}unclean_tables/{self.name}.csv')

    def _write_query_for_features(self):
        print('Writing SQL query for features...', flush= True)
        # open and read file with columns
        file_cols = open(f'{self.dir}/SQL_queries/columns.txt')
        cols = file_cols.read()
        file_cols.close()

        query = (" WITH ids(object_id) as (values "
            "({}) "
            ") "
            "SELECT "
            "{} "
            "FROM pdr2_wide.forced "
            "LEFT JOIN pdr2_wide.forced2 USING (object_id) "
            "LEFT JOIN pdr2_wide.forced3 USING (object_id) "
            "NATURAL JOIN ids"
            )

        with open(f'{self.dir}SQL_queries/{self.name}_feat.sql', mode = 'w') as s:
            s.write(query.format('),('.join(map(str, self.df_gal.object_id.values)), cols))

    def _run_query_for_features(self):
        print('Running SQL query for features...', flush= True)
        import os
        command = f'python hscReleaseQuery.py --user {self.user} --format fits --delete-job SQL_queries/{self.name}_feat.sql > SQL_queries/{self.name}_feat.fits '
        os.system(command)

    def _match_results(self):
        print('Matching results and saving...', flush= True)
        from astropy.table import Table

        # read fits
        df_sql = Table.read(f'{self.dir}SQL_queries/{self.name}_feat.fits', format='fits')
        df_sql = df_sql.to_pandas()

        # join
        self.df_gal = pd.merge(left = self.df_gal, right = df_sql, on = 'object_id', how = 'left')
        self.df_gal.to_csv(f'{self.dir}unclean_tables/{self.name}.csv', index=False)