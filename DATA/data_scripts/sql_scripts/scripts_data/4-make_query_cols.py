"""
Write an SQL query that selects a set of columns on a set of ids.

From command line: python make_query_cols.py HSC-unWISE-W01
Argument: name of the csv file with the galaxies ids (without extension)

Columns names are in a text file (cols.txt), separated by commas.
"""

import pandas as pd
import sys

file = sys.argv[1]
df_gal = pd.read_csv(f'3-joined-mem-gal/{file}.csv')

# open and read file with columns
file_cols = open('cols.txt')
cols = file_cols.read()
file_cols.close()

# sql query
# query = ("SELECT "
#          "{} "
#          "FROM pdr2_wide.forced "
#          "LEFT JOIN pdr2_wide.forced2 USING (object_id) "
#          "LEFT JOIN pdr2_wide.forced3 USING (object_id) "
#          "WHERE object_id in ({})")

# with open('3-sql-cols/{}.sql'.format(file[:-4]), mode = 'x') as s:
#     s.write(query.format(cols, ','.join(map(str, df_gal.object_id.values))))

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

with open('4-sql-cols/{}.sql'.format(file), mode = 'x') as s:
    s.write(query.format('),('.join(map(str, df_gal.object_id.values)), cols))