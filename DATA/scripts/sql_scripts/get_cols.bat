set arg=%1

cd scripts_and_data

python 4-make_query_cols.py %arg%

python hscReleaseQuery.py --user gimenam144 --format fits --delete-job 4-sql-cols/%arg%.sql > 4-sql-cols/%arg%.fits

python 5-match_results.py %arg%

cd OUT

move %arg%.csv ../../

cd ../../




