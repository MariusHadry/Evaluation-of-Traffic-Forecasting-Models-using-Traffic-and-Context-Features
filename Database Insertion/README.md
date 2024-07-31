# Insertion of Data into a PostgreSQL Database


## Prerequisites
Before executing the scripts to insert the traffic data into the database, the routable database should already be created.

## Script Execution Order
1. Run `OpenlrExtraction.py`
    + Modify the path to the zip files (`path_to_zip_files` variable)
    + Modify database connection information (`connection` variable)
2. Resolve OLR-Codes and create the table for olr-eid mapping
3. Run `ParseTrafficDataToDatabase.py`
   + Modify the path to the zip files (`path_to_zip_files` variable)
   + Modify database connection information (`connection` variable)
4. Run `ParseWeatherDataToDatabase.py`
   + Modify the path to the zip files (`path_to_zip_files` variable)
   + Modify database connection information (`connection` variable)