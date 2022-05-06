# join police_short with listings.csv
####### Dat: 04/17
# imports:
from matplotlib import pyplot as plt  
import pandas as pd  
import numpy as np  
from sklearn.preprocessing import StandardScaler  


# Getting the data
police = spark.read.options(header ='True',inferSchema='True',delimiter=',').csv("team3/etl_code/data.csv")  
police = police.toPandas()  
air = spark.read.options(header ='True',inferSchema='True',delimiter=',').csv("team3/etl_code/airbnb.csv")  
air = air.toPandas()  

# data preprocessing
police["latitude"] = police["Y"]  
police["longitude"] = police["X"]  
police["tmp"] = 1  
air["tmp"] = 1  
air_brief = air[["id", "latitude","longitude",'tmp']]  
police_brief = police[["PdId", "latitude","longitude",'tmp']]  

# join: create a temp table that is the result of cross join. 
temp = pd.merge(air_brief,police_brief,on='tmp')  

# calculates the distance
def distance(x1, x2, y1, y2):  
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)  
    
temp["distance"] = distance(temp["latitude_x"],temp["latitude_y"],temp["longitude_x"], temp["longitude_y"])  

# select the rows where the locations are closer, distance <=0.05
temp1= temp[temp["distance"]<=0.05]  

# count the number of crimes within the 0.05 radius and add a column to air df
gg = temp1.groupby('id').count()  
k = gg["distance"]  
kk = k.value_counts()  
plt.hist(kk)  
scheme = gg[["distance"]]  
scheme = scheme.to_dict()["distance"]  
crime_counts = [scheme[i] if i in scheme else 0 for i in air["id"]] # distance to crime location < 0.05  
air["crime_counts"] = crime_counts  

# find the closet crime location and join with the air table
g = temp.groupby('id')  


# create an intermediate mapping table
closest = g.apply(lambda x: x.sort_values(by=["distance"],ascending=True).head(1))  
mapping = closest[['id','PdId']]  
mapping = mapping.set_index(["id"])  
mapping = mapping.to_dict()["PdId"]  
air["PdId"] = [mapping[i] for i in air["id"]]  
airbnb_police = pd.merge(air, police, on="PdId")  


# clean the dataset
#airbnb_police = airbnb_police.drop(['X', 'Y',"tmp_x",'Unnamed: 0','tmp_y'], axis = 1)  
airbnb_police = airbnb_police.drop(['X', 'Y',"tmp_x",'tmp_y'], axis = 1)  
airbnb_police["distance"] = distance(airbnb_police["latitude_x"], airbnb_police["latitude_y"], airbnb_police["longitude_x"], airbnb_police["longitude_y"])  
airbnb_police = airbnb_police.rename(index={"PdId": "closest crime PdId", "crime_counts": "crime_counts < 0.05"})  
airbnb_police["price"] = airbnb_police["price"].str.replace(",","")  
airbnb_police.rename(columns={"PdId": "closest crime PdId", "crime_counts": "crime_counts < 0.05"},inplace = True)  

# some analysis-correlation
crime_price = airbnb_police[["crime_counts < 0.05","price"]]  
scaler = StandardScaler()  
scaler.fit(crime_price)  
new_crime_price = scaler.transform(crime_price) 
fig = plt.figure()
plt.scatter(new_crime_price[:,0], new_crime_price[:,1])  
print(np.corrcoef(new_crime_price[:,0], new_crime_price[:,1]))

fig.savefig('crime_number vs price.png')
airbnb_police.to_csv("team3/etl_code/airbnb_police.csv", index=False) # write to home  



