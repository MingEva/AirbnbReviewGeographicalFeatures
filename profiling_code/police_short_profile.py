import numpy as np
import pandas as pd

def na_profile(df):
    print('number of rows: ',len(df))
    print('number of rows with any nan:')
    actual_na=df.shape[0] - df.dropna().shape[0]
    print(actual_na)
    print('null_percentage of rowss: ',actual_na/len(df)*100)
    print('columns and their numbers of nans:')
    print(df.isna().sum())


df = spark.read.options(header ='True',inferSchema='True',delimiter=',').csv("team3/data_ingest/police_short.csv")  
df = df.toPandas()
na_profile(df)

for col in df.columns:
	print(col)
	print(df[col].value_counts(dropna = False))
	print(df[col].describe())