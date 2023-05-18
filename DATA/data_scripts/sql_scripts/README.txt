--------------
1---
run make_query_ids.py to write SQL queries for getting ids + coords

-------------
2---
get_ids_and_match.bat
submits a query, matches results and join members and galaxy tables.
Argument: file name without extension (e.g.: HSC-unWISE-W01)

------------
3---
get_cols.bat
writes a query to select columns for certain ids, submits it, and joins results to original table. Moves output to this directory.
Columns to be selected are in cols.txt
Argument: file name without extension (e.g.: HSC-unWISE-W01)
