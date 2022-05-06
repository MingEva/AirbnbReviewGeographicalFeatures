
#------------------------------------------------CODE--------------------------------------

#PATHS
input_business='team3/business_cleaned.csv'
input_airbnb_police='team3/airbnb_police.csv'
output_path='team3/final_join.csv'


#input of cleaned
business=spark.read.options(header='True',inferSchema='True',delimiter=',').csv(input_business)
business=business.toPandas()

airbnb_police=spark.read.options(header='True',inferSchema='True',delimiter=',').csv(input_airbnb_police)
airbnb_police=airbnb_police.toPandas()

import pandas as pd
import numpy as np

business['tmp']=1
airbnb_police['tmp']=1

# join: create a temp table that is the result of cross join. 
temp = pd.merge(airbnb_police,business,on='tmp')

temp=temp.rename(columns={"longitude": "longitude_z", "latitude": "latitude_z"}) 
# calculates the distance
def distance(x1, x2, y1, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
temp["distance"] = distance(temp["latitude_x"],temp["latitude_z"],temp["longitude_x"], temp["longitude_z"])

# select the rows where the locations are closer, distance <=0.05
temp1= temp[temp["distance"]<=0.05]

# count the number of crimes within the 0.05 radius and add a column to air df
gg = temp1.groupby('id').count()
k = gg["distance"]
kk = k.value_counts()


scheme = gg[["distance"]]
scheme = scheme.to_dict()["distance"]
# distance to crime location < 0.05
business_counts = [scheme[i] if i in scheme else 0 for i in airbnb_police["id"]] 
airbnb_police["business_counts"] = business_counts

# find the closet crime location and join with the air table
g = temp.groupby('id')
# create an intermediate mapping table
closest = g.apply(lambda x: x.sort_values(by=["distance"],ascending=True).head(1))
mapping = closest[['id','Location Id']]
mapping = mapping.set_index(["id"])
mapping = mapping.to_dict()["Location Id"]
airbnb_police["Location Id"] = [mapping[i] for i in airbnb_police["id"]]
final = pd.merge(business, airbnb_police, on="Location Id")

#encode
df=final.copy()

#Encode columns
df.loc[df['Parking Tax'] == False, 'Parking Tax'] = 0
df.loc[df['Parking Tax'] == True, 'Parking Tax'] = 1
df.loc[df['Transient Occupancy Tax'] == False, 'Transient Occupancy Tax'] = 0
df.loc[df['Transient Occupancy Tax'] == True, 'Transient Occupancy Tax'] = 1
 
#convert data types of new encoded columns to ‘float’ for multiple regression to successfully run later
df['Transient Occupancy Tax'] = df['Transient Occupancy Tax'].astype('float64')
df['Parking Tax'] = df['Parking Tax'].astype('float64')

#check datatypes of columns
df.dtypes


#data profiling
def na_profile(df):
    print('number of rows: ',len(df))
    print('number of rows with any nan: ',)

    actual_na=df.shape[0] - df.dropna().shape[0]

    print(actual_na)

    print('null_percentage: ',actual_na/len(df)*100)

na_profile(df)

#dropna
df=df.dropna()

print('post-dropping null values:')
na_profile(df)

#save output data to csv
df=spark.createDataFrame(df)
df.coalesce(1).write.format('com.databricks.spark.csv').options(header='true').save(output_path)
