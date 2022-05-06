
# clean poilce_short.csv
import pandas as pd  
import numpy as np 
police = spark.read.options(header ='True',inferSchema='True',delimiter=',').csv("team3/data_ingest/police_short.csv")  
police = police.toPandas()  

# clean my data
# police = police.drop(["_c0"],axis = 1)  

# calculate the mean, percentiles of my data 
police.describe()  

# to see the distribution of crimes, I group by police District ID
police.groupby(by="PdId").sum()  

# to enable comparison with other datasets, I group by police district string
police.groupby(by="PdDistrict").sum()  
temp = police.groupby(by="PdDistrict").sum()

# I create a new column for the sum of crimes in each district
police["district_total_crimes"] = police["PdDistrict"].map(lambda x: temp["IncidntNum"][x])  

# I export the data to home on peel 
police.to_csv("data.csv", index=False)  

# after this code, I've changed the location of data in hdfs: in furture to generate the data.csv, use  project/new_data/data.csv, to access the original police data, use project/old_data/police_short.csv
