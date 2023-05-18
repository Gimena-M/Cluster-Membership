set arg=%1

cd scripts_data

python hscReleaseQuery.py --user gimenam144 --format fits --delete-job "1-sql-ids/%arg%.sql" > "1-sql-ids/%arg%.fits"

python 2-matching.py %arg%

python 3-join_mem.py %arg%

cd ..


