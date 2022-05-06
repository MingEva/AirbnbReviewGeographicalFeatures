#--------------code-------------------
#input
input_path='team3/business.csv'
output_path="team3/business_cleaned.csv"

#read data
data=spark.read.options(header='True',inferSchema='True',delimiter=',').csv(input_path)
data=data.toPandas()

import pandas as pd
import re


#profiling nans
def na_profile(df):
    print('number of rows: ',len(df))
    
    print('number of rows with any nan:')
    actual_na=df.shape[0] - df.dropna().shape[0]
    print(actual_na)

    print('null_percentage of rowss: ',actual_na/len(df)*100)
    
    print('columns and their numbers of nans:')
    print(df.isna().sum())
    

print('oroginal dataset: ')
na_profile(data)
print()

#select the columns we want to use
df=data[['Location Id','DBA Name',
        'Source Zipcode',
        'Neighborhoods - Analysis Boundaries',
        'Business Location',
        'Parking Tax',
        'Transient Occupancy Tax']]
 

print('dataset after selecting columns: ')
na_profile(df)
print()


      
df=df.dropna()

#change longitude and latitude
df['latitude']=df['Business Location'].apply(lambda x:re.findall(r"[-+]?(?:\d*\.\d+|\d+)",x)[1])
df['longitude']=df['Business Location'].apply(lambda x:re.findall(r"[-+]?(?:\d*\.\d+|\d+)",x)[0])

df['latitude']=pd.to_numeric(df['latitude'])
df['longitude']=pd.to_numeric(df['longitude'])

# delete small
business=df.copy()
business.reset_index(inplace=True,drop=True)

business

samp=business.sample(6413,random_state=0).reset_index(drop=True)

#output business
df=spark.createDataFrame(samp)
df.coalesce(1).write.format('com.databricks.spark.csv').options(header='true').save(output_path)

