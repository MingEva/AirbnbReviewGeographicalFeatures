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
#numerical data analysis
col = ['latitude_x', 'longitude_x', 'price', 'number_of_reviews', 'review_scores_rating', 'reviews_per_month', 'crime_counts', 'distance']
price_corr = {}
rating_corr = {}
num_rev_cor = {}
for c in col:
    price_corr[c] = joint[[c,'price']].corr(method="spearman").iloc[1,0]
    rating_corr[c] = joint[[c,'review_scores_rating']].corr(method="spearman").iloc[1,0]
    num_rev_cor[c] = joint[[c,'number_of_reviews']].corr(method="spearman").iloc[1,0]
##################################################################################################################
index = pd.Index(['price','rating','num_reviews'])
correlations = pd.DataFrame.from_dict([price_corr, rating_corr, num_rev_cor]).set_index(index)
with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    print(correlations)

##################################################################################################################
cat_col = ['neighbourhood_cleansed', 'latitude_x', 'longitude_x','Parking Tax', 'Transient Occupancy Tax']
for c in cat_col:
    print(f'{c}:')
    print(joint.groupby(c).size().sort_values(ascending = False),'\n')


##################################################################################################################
# a list of dataframes, each is a borough
gb = joint.groupby('neighbourhood_cleansed') 
borough_data = [gb.get_group(x) for x in gb.groups]

#numeric values and correlation

bor_corr = {}

for bor in borough_data:
  col = ['latitude_x', 'longitude_x', 'price', 'number_of_reviews', 'review_scores_rating',
       'reviews_per_month', 'crime_counts',
       'distance']
  price_corr = {}
  rating_corr = {}
  num_rev_cor = {}
  for c in col:
      print(f'{c}:')
      print(bor[c].describe()[1:],'\n')
      print(f'{c} mode:',bor[c].mode()[0],'\n')
      price_corr[c] = bor[[c,'price']].corr(method="spearman").iloc[1,0]
      rating_corr[c] = bor[[c,'review_scores_rating']].corr(method="spearman").iloc[1,0]
      num_rev_cor[c] = bor[[c,'number_of_reviews']].corr(method="spearman").iloc[1,0]
  index = pd.Index(['price','rating','num_reviews'])
  bor_corr[bor['neighbourhood_cleansed'].iloc[0]] = pd.DataFrame.from_dict([price_corr, rating_corr, num_rev_cor]).set_index(index)
##################################################################################################################

for i in bor_corr:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
      print(i, bor_corr[i], '\n')
##################################################################################################################


bor_price_mean = {}
bor_numrev_mean = {}
bor_crime_mean = {}
bor_rating_mean = {}
bor_dis_mean = {}

col = ['price', 'number_of_reviews', 'review_scores_rating', 'crime_counts', 'distance']
for c in col:
  for bor in borough_data:
    b = bor['neighbourhood_cleansed'].iloc[0]
    if c == 'price':
      bor_price_mean[b] = np.mean(bor[c])
    if c == 'number_of_reviews':
      bor_numrev_mean[b] = np.mean(bor[c])
    if c == 'review_scores_rating':
      bor_crime_mean[b] = np.mean(bor[c])
    if c == 'crime_counts':
      bor_rating_mean[b] = np.mean(bor[c])
    if c == 'distance':
      bor_dis_mean[b] = np.mean(bor[c])
##################################################################################################################

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

for dic in [bor_price_mean, bor_numrev_mean, bor_crime_mean, bor_rating_mean, bor_dis_mean]:
  print(namestr(dic, globals())[0], sorted(dic.items(), key=lambda item: item[1], reverse=True))
