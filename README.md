# Telling Fortunes? Evaluation of Traffic Forecasting Models using Traffic and Context Features

The overall workflow is:

1. Collect data using the crawler
2. Create database using osm2psql with the open street map data that fits the area chosen in the crawler
3. Insert raw data into database
4. Decode OpenLR codes of the raw data
5. Use Data Fusion Tool to create training data in the form of *.csv files
6. Train and evaluate models using the code in `Baselines/` and `DeepLearning/`

![image](images/APIN%20Workflow-Data.jpg)
