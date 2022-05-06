import pandas as pd
import numpy as np

##################################################################################################################
#build dataset
joint = spark.read.options(header ='True',inferSchema='True', delimiter=',').csv("team3/final_join.csv")
joint.printSchema()
joint = joint.toPandas()
print(joint)

##################################################################################################################
#preprocess data, choose the ones we need
joint = joint[['neighbourhood_cleansed', 'latitude_x', 'longitude_x', 'price', 'number_of_reviews', 'review_scores_rating',
       'reviews_per_month', 'crime_counts',
       'distance', 'Parking Tax', 'Transient Occupancy Tax']]

joint.loc[joint['Parking Tax'] == False, 'Parking Tax'] = 0
joint.loc[joint['Parking Tax'] == True, 'Parking Tax'] = 1

joint.loc[joint['Transient Occupancy Tax'] == False, 'Transient Occupancy Tax'] = 0
joint.loc[joint['Transient Occupancy Tax'] == True, 'Transient Occupancy Tax'] = 1


joint['Transient Occupancy Tax'] = joint['Transient Occupancy Tax'].astype('float64')
joint['Parking Tax'] = joint['Parking Tax'].astype('float64')

joint.dropna(inplace = True)

##################################################################################################################
# Datatype observed always (or almost always) in the column
print(joint.dtypes)

##################################################################################################################
#numerical data count, min, max, mean
col = ['latitude_x', 'longitude_x', 'price', 'number_of_reviews', 'review_scores_rating', 'reviews_per_month', 'crime_counts', 'distance']
for c in col:
    print(f'{c}:')
    print(joint[c].describe()[1:],'\n')
    print(f'{c} mode:',joint[c].mode()[0],'\n')

##################################################################################################################
#categorical columns, unique values & count
cat_col = ['neighbourhood_cleansed', 'latitude_x', 'longitude_x','Parking Tax', 'Transient Occupancy Tax']
for c in cat_col:
    print(f'{c}:')
    print(joint.groupby(c).size().sort_values(ascending = False),'\n')

