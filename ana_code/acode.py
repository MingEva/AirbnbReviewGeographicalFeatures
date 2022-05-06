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
# X_price.to_csv("team3/etl_code/X_price.csv", index=False)  
y_price = y
# y_price.to_csv("team3/etl_code/y_price_real.csv", index=False) 
# showed X_price
X_price

X = a4.drop(["review","review_scores_rating"], axis = 1)
X = zero_one(['Parking Tax', 'Transient Occupancy Tax'],X)
X = zscore(['number_of_reviews', 'reviews_per_month','crime_counts', 'distance'], X)
X_review = X
# X_review.to_csv("team3/etl_code/X_review.csv", index=False) 
y_review_bi = a4["review"]
# y_review_bi.to_csv("team3/etl_code/y_review_bi.csv", index=False) 
# showed X_review
X_review












# linear regression for price as a target
X_price = spark.read.options(header ='True',inferSchema='True',delimiter=',').csv("team3/etl_code/X_price.csv")  
X_price = X_price.toPandas()
X_price = X_price.drop(["id"], axis = 1)
y_price = spark.read.options(header ='True',inferSchema='True',delimiter=',').csv("team3/etl_code/y_price_real.csv")  
y_price = y_price.toPandas()
X_price["price"] = y_price
X = X_price
from pyspark.ml.feature import VectorAssembler
df = sqlContext.createDataFrame(X)
assembler = VectorAssembler(inputCols=['number_of_reviews', 'review_scores_rating', 'reviews_per_month',
       'crime_counts', 'distance', 'Parking Tax', 'Transient Occupancy Tax',
       'review'],outputCol='features')
trainingData = assembler.transform(df)
trainingData.show()
df = trainingData
df = df.select(['features','price'])
df.show()


from pyspark.ml.regression import LinearRegression
splits = df.randomSplit([0.8,0.2])
train_df = splits[0]
test_df = splits[1]
lr = LinearRegression(featuresCol = 'features', labelCol='price', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)



# random forest
X_review = spark.read.options(header ='True',inferSchema='True',delimiter=',').csv("team3/etl_code/X_review.csv")  
X_review = X_review.toPandas()
X_review = X_review.drop(["id"], axis = 1)
y_review = spark.read.options(header ='True',inferSchema='True',delimiter=',').csv("team3/etl_code/y_review_bi.csv")  
y_review = y_review.toPandas()
X_review["review"] = y_review
X = X_review


from pyspark.ml.feature import VectorAssembler
df = sqlContext.createDataFrame(X)
assembler = VectorAssembler(inputCols=['price', 'number_of_reviews', 'reviews_per_month', 'crime_counts',
       'distance', 'Parking Tax', 'Transient Occupancy Tax', 'price_rating'],outputCol='features')
trainingData = assembler.transform(df)
df = trainingData
df = df.select(['features','review'])
df.show()



train, test = df.randomSplit([0.8, 0.2], seed = 2018)
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'review')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions.show(25)




from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="review", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % (accuracy))
print("Test Error = %s" % (1.0 - accuracy))











# pca and dropone out


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
def general_dropone_predictor(XTrain, XTest, yTrain, yTest, original_model): 
    drop_times = XTrain.shape[1] # 21
    print(f"what is drop times: {drop_times}")
    original_XTrain = XTrain
    original_XTest = XTest
    container_test = []
    for i in range(0, drop_times+1):  # +1 because last row we want all predictors (0-21)
        XTrain = original_XTrain # restore XTrain
        XTest = original_XTest 
        if i < drop_times:
            print(f"Log: excluded column = {XTest.columns[i]}. index={i}")
            XTrain = XTrain.drop(XTrain.columns[i], axis =1)
            XTest = XTest.drop(XTest.columns[i], axis =1)
            total_number_of_predictors = XTrain.shape[1]
            # train, fit
            yTrain = np.asarray(yTrain)
            model = original_model 
            model.fit(XTrain,yTrain)
            # test set, predict
            yTest = np.asarray(yTest)
            y_pred = model.predict(XTest)
            y_pred_prob = model.predict_proba(XTest)
            accu = metrics.accuracy_score(yTest,y_pred)
            recall = metrics.recall_score(yTest,y_pred)
            precision = metrics.precision_score(yTest,y_pred)
            f1 = metrics.f1_score(yTest,y_pred)
            auc = metrics.roc_auc_score(y_test,y_pred_prob[:, 1])
            container_test.append([original_XTest.columns[i], auc, accu, recall, precision, f1])
        # full multiple LR
        elif i==drop_times: 
            # train
            yTrain = np.asarray(yTrain)
            model = original_model 
            model.fit(XTrain,yTrain)
            yTest = np.asarray(yTest)
            y_pred = model.predict(XTest)
            y_pred_prob = model.predict_proba(XTest)
            accu = metrics.accuracy_score(yTest,y_pred)
            recall = metrics.recall_score(yTest,y_pred)
            precision = metrics.precision_score(yTest,y_pred)
            f1 =  metrics.f1_score(yTest,y_pred)
            auc = metrics.roc_auc_score(y_test,y_pred_prob[:, 1])
            container_test.append(["Full Multiple Regression", auc, accu, recall, precision, f1])
    df2 = pd.DataFrame(container_test, columns = ["Excluded Predictor Name", 'auc',"Accuracy", "Recall", "Precision","f1"])
    df2.sort_values(by= ["auc"], inplace = True)
    full_score = df2[df2["Excluded Predictor Name"]=="Full Multiple Regression"]
    print("On Test Set\n", df2, "\n")
    return (full_score, df2)
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
a4 = spark.read.options(header ='True',inferSchema='True',delimiter=',').csv("project/final_join.csv")  
a4 = a4.toPandas()

des = a4.drop(["id", "Incident Code", "Source Zipcode", 'latitude_y', 'longitude_y',"IncidntNum","PdId",  "latitude_x", "longitude_x", "Date","Address", "location", "DBA Name", "Business Location",  "latitude_y", "longitude_y", 'longitude_z29','longitude_z30'], axis=1)
for col in des.columns:
    print(col)
    print(des.groupby(des[col]).describe())
    continue   
 
 # very long output: only one picture selected







# pcas

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
# screen shot 1


y = a4["review"]
X = a4.drop(["review","review_scores_rating"], axis = 1)
X = zero_one(['Parking Tax', 'Transient Occupancy Tax'],X)
X = zscore(['number_of_reviews', 'reviews_per_month','crime_counts', 'distance'], X)
X = X.drop(["id"], axis =1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)


def full_model(X_train,y_train,X_test, y_test, model, model_label):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    accu = metrics.accuracy_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    precision = metrics.precision_score(y_test,y_pred)
    f1 = metrics.f1_score(y_test,y_pred)
    auc = metrics.roc_auc_score(y_test,y_pred_prob[:, 1])
    report = [accu, recall, precision, f1, auc]
    report = pd.Series(report, ['accu','recall','precision','f1', 'auc'])
    print(f"Full model performance: \n{report}")
    print(f"{model_label} Graph:")
    lr_fpr, lr_tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred_prob[:, 1])
    fig = plt.figure()
    plt.plot(lr_fpr, lr_tpr, marker='.', label=model_label)
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC_AUC Curve of the Logistic Regression Model")
    plt.show()
    fig.savefig('ROC_AUC.png')
    print(thresholds)
    return (y_pred, y_pred_prob, lr_fpr, lr_tpr)



model = RandomForestClassifier(n_estimators=100, max_samples=1.0, max_features=0.5,bootstrap=True, criterion='gini')
y_pred, y_pred_prob,lr_fpr_4, lr_tpr_4 = full_model(X_train,y_train,X_test, y_test, model, "Random Forest")

pca = PCA(n_components=8, whiten=True)
pca.fit(X)
X_pca = pca.transform(X)
target_ids = range(len(y))
fig = plt.figure(figsize=(5, 5))

for i, c, label in zip(target_ids, 'rgbcmykw', [0,1]):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], c=c, label=label)

plt.xlabel('PC5')
plt.ylabel('PC6')
plt.title("Airbnb Review Score PCA")
plt.legend()
plt.show()
fig.savefig('Airbnb_Review_Score_PCA.png')


fig = plt.figure()
per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
labels = ["PC" + str(x) for x in range(1, len(per_var)+1)]
plt.bar(x = range(1, len(per_var)+1), height = per_var, tick_label = labels)
plt.title("PCA Variance Explained")
fig.savefig('PCA_variance_dis.png')


loading_scores = pd.Series(pca.components_[0], index = X.columns)
sorted_loading_scores = loading_scores.abs().sort_values(ascending = False)
top_5 = sorted_loading_scores[0:5].index.values
print(loading_scores[top_5])
loading_scores = pd.Series(pca.components_[1], index = X.columns)
sorted_loading_scores = loading_scores.abs().sort_values(ascending = False)
top_5 = sorted_loading_scores[0:5].index.values
print(loading_scores[top_5])
print(pca.components_.T * np.sqrt(pca.explained_variance_))

