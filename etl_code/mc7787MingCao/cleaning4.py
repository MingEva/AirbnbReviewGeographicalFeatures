from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.stats as stats
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
import sklearn
pd.options.display.max_columns = 50
def target_encoding(target_column, column_to_transform, trainset, X_train, y_train, X_test, y_test): 
    # find the mapping based on the train set
    mapping = trainset.groupby(trainset[column_to_transform]).mean()
    mapping = mapping[target_column]
    # transfor X_train
    new_train_column = [mapping[mapping.index==i] for i in X_train[column_to_transform]]
    li_train = []
    for i in new_train_column:
        try:
            li_train.append(i[0])
        except:
            li_train.append(0)
    # transform X_test
    new_test_column = [mapping[mapping.index==i] for i in X_test[column_to_transform]]
    li_test = []
    for j in new_test_column:
        try: 
            li_test.append(j[0])
        except: 
            li_test.append(0)
    return li_train,li_test, mapping
def target_encode_all(target_column, columns, X_train, y_train, X_test, y_test): 
    temp_X_train = X_train.copy()
    temp_X_train[target_column] = y_train
    trainset = temp_X_train
    new_X_train = X_train.copy()
    new_X_test = X_test.copy()
    for column in columns:
        t = target_encoding(target_column, column, trainset, X_train, y_train, X_test, y_test)
        new_train_column = t[0]
        new_test_column = t[1]
        new_X_train[column] = new_train_column
        new_X_test[column] = new_test_column
    return new_X_train, new_X_test
def zero_one(columns, df): 
    dfc = df.copy()
    for col in columns:
        new_col = [1 if i==True else 0 for i in df[col]]
        dfc[col] = new_col
    return dfc
def min_max(columns, df):
    dfc = df.copy()
    for col in columns:
        std_scale = preprocessing.MinMaxScaler().fit(np.asarray(dfc[col]).reshape(-1, 1))
        dfc[col] = std_scale.transform(np.asarray(df[col]).reshape(-1, 1))
    return dfc
def zscore(columns, df):
    dfc = df.copy()
    for col in columns:
        dfc[col] = stats.zscore(df[col])
    return dfc
target = "review_scores_rating"
a4 = spark.read.options(header ='True',inferSchema='True',delimiter=',').csv("team3/etl_code/final_join.csv")  
a4 = a4.toPandas()
drop_columns = ["Incident Code", "Source Zipcode", 'host_neighbourhood', 'Category', 'Descript', 'Neighborhoods - Analysis Boundaries', 'neighbourhood_cleansed','latitude_y', 'longitude_y',"IncidntNum","PdId",  "latitude_x", "longitude_x", "Date","PdDistrict","Address", "location", "DBA Name", "Business Location", "DayOfWeek", "latitude_y", "longitude_y", 'longitude_z29','longitude_z30']
a4 = a4.drop(drop_columns, axis =1)
a4 = a4.dropna()
cate_price= []
for i in a4.price:
    if i<95.000000:
        cate_price.append(0) # LOW PROCE
    elif i>=95 and i<144:
        cate_price.append(1)
    elif i>=144 and i<225:
        cate_price.append(2)
    else:
        cate_price.append(3) # high price
cate_rating=[]
for i in a4[target]:
    if i<4.9:
        cate_rating.append(0) # LOW rating
    else:
        cate_rating.append(1) # high rating
a4["review"] = cate_rating
a4["price_rating"] = cate_price

# I have an output here showing a4
a4


y = a4["price"]
X = a4.drop(["price", "price_rating"], axis = 1)
X = zero_one(['Parking Tax', 'Transient Occupancy Tax'],X)
X = zscore(['number_of_reviews', 'reviews_per_month','crime_counts', 'distance'], X)
X_price = X
X_price.to_csv("team3/etl_code/X_price.csv", index=False)  
y_price = y
y_price.to_csv("team3/etl_code/y_price_real.csv", index=False) 
# showed X_price
X_price

X = a4.drop(["review","review_scores_rating"], axis = 1)
X = zero_one(['Parking Tax', 'Transient Occupancy Tax'],X)
X = zscore(['number_of_reviews', 'reviews_per_month','crime_counts', 'distance'], X)
X_review = X
X_review.to_csv("team3/etl_code/X_review.csv", index=False) 
y_review_bi = a4["review"]
y_review_bi.to_csv("team3/etl_code/y_review_bi.csv", index=False) 
# showed X_review
X_review


