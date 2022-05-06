airbnb = spark.read.options(header ='True',inferSchema='True', delimiter=',').option("multiLine",'True').option("escape","\"").csv("team3/listings.csv")

airbnb.printSchema()
airbnb = airbnb.toPandas()
airbnb = airbnb[['id',	'host_neighbourhood',	'neighbourhood_cleansed',	'latitude',	'longitude',	'price','number_of_reviews',	'review_scores_rating',	'reviews_per_month']]
import pandas as pd
'''with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    print(airbnb)'''

beforedrop = len(airbnb)
print(len(airbnb))
airbnb.dropna(inplace = True)
afterdrop = len(airbnb)
print(len(airbnb))
print('Nan Percentage:', (beforedrop-afterdrop)/beforedrop)

#calculate the descriptive statistics and calculate correlation coefficients.

airbnb.iloc[:,5] = airbnb.iloc[:,5].str.replace('$', '').str.replace(',', '').astype(float)
print(airbnb.dtypes)


airbnb=spark.createDataFrame(airbnb)
airbnb.coalesce(1).write.format('com.databricks.spark.csv').options(header='true').save("team3/airbnb.csv")
