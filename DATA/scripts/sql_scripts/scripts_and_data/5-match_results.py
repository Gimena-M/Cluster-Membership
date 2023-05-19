"""
Match results from the SQL query with the galaxies table, to add extra features.
From command line : python match_results.py HSC-unWISE-W01 
Arguments: name of the .csv file with galaxies, and of the .fits file with the query results (without extension)
"""

import pandas as pd
from astropy.table import Table
import sys

file = sys.argv[1]

# read csv
df_gal = pd.read_csv('3-joined-mem-gal/' + file + '.csv')

# read fits
dat = Table.read('4-sql-cols/'+ file + '.fits', format='fits')
df_sql = dat.to_pandas()
del dat

# join
a = pd.merge(left = df_gal, right = df_sql, on = 'object_id', how = 'left')
a.to_csv('OUT/' + file + '.csv', index=False)