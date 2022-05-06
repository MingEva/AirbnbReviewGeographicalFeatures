# Readme.md

If the script doesn’t work, please copy the code directly into the spark shell. 

If a file exists, and you would like to see it generated for the first time, please remove it. Thank you!

## Data Ingest

### Download Data

1. Download San Francisco Airbnb Review & Pricing Data: 
    1. Go to http://insideairbnb.com/get-the-data/
    2. Find “San Francisco, California, United States”
    3. Download the first file in the list, “[listings.csv.gz](http://data.insideairbnb.com/united-states/ca/san-francisco/2021-12-04/data/listings.csv.gz)”, and unzip the file locally to get `listings.csv`
2. Download police_short.csv  (the first 6000 rows)
    1. go to https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry
    2. download and rename as police_short.csv
3. Download business.csv
    1. Go to https://data.sfgov.org/Economy-and-Community/Registered-Business-Locations-San-Francisco/g8m3-pdis
    2. click “export”, click “csv”
    3. rename the file to `business.csv`
    

### Upload Data

- Please change it according to your home directory:
- `scp /Users/guozhitong/Downloads/listings.csv zg978@peel.hpc.nyu.edu:/home/zg978/`
- `scp /Users/guozhitong/Downloads/business.csv zg978@peel.hpc.nyu.edu:/home/zg978/`
- `scp /Users/guozhitong/Downloads/police_short.csv zg978@peel.hpc.nyu.edu:/home/zg978/`
- `hdfs dfs -put police_short.csv team3`
- `hdfs dfs -put business.csv team3`
- `hdfs dfs -put listings.csv team3`

## ETL - Clean data with PySpark

### listings.csv:

- run `scp /Users/guozhitong/Downloads/clean_airbnb.py zg978@peel.hpc.nyu.edu:/home/zg978`
- run `hdfs dfs -rm -r team3/airbnb.csv`
- run `module load python/gcc/3.7.9`
- run `PYTHONSTARTUP=clean_airbnb.py pyspark --deploy-mode client`
- The output has been stored as `airbnb.csv`

### police_short.csv:   please adjust the file paths as you see fit on your computer

1. Upload the cleaning2.py to clean the police_short.csv
    
    `scp /Users/caoming/malavet/project/cleaning2.py  mc7787@peel.hpc.nyu.edu:/home/mc7787/team3/etl_code`
    
    `hdfs dfs -put team3/cleaning2.py team3`
    
2. Run cleaning2.py in home on peel, move the output data.csv into team3 folder on peel and also upload it onto hdfs
    
    `module load python/gcc/3.7.9`
    
    `PYTHONSTARTUP=team3/etl_code/cleaning2.py pyspark --deploy-mode client`
    
    `mv data.csv team3`
    
    `hdfs dfs -put team3/data.csv team3`
    
3. Upload cleaning3.py
    1. `scp /Users/caoming/malavet/project/cleaning3.py mc7787@peel.hpc.nyu.edu:/home/mc7787/team3/etl_code`
    2. `hdfs dfs -put team3/cleaning3.py team3`
    3. `module load python/gcc/3.7.9`
    4. `PYTHONSTARTUP=team3/etl_code/cleaning3.py pyspark --deploy-mode client`
4. upload the resulting airbnb_police.csv to hdfs
    1. `hdfs dfs -put team3/etl_code/airbnb_police.csv team3/etl_code`
    2. `scp mc7787@peel.hpc.nyu.edu:/home/mc7787/team3/etl_code /Users/caoming/malavet/project/cleaning3.py` 
5. upload cleaning4.py and run:
    1. `scp /Users/caoming/malavet/project/cleaning4.py mc7787@peel.hpc.nyu.edu:/home/mc7787/team3/etl_code`
    2. `PYTHONSTARTUP=team3/etl_code/cleaning4.py pyspark --deploy-mode client`
    3. output: X_price.csv X_review.csv y_price_real.csv y_review_bi.csv 
    4. upload the output csvs to hdfs
        1. hdfs dfs -put team3/etl_code/X_price.csv team3/etl_code
        2. hdfs dfs -put team3/etl_code/X_review.csv team3/etl_code
        3. hdfs dfs -put team3/etl_code/y_price_real.csv team3/etl_code
        4. hdfs dfs -put team3/etl_code/y_review_bi.csv team3/etl_code
        

### business.csv cleaning :

- run `hdfs dfs -rm -r team3/business_cleaned.csv`
- run `module load python/gcc/3.7.9`
- run `PYTHONSTARTUP=business_cleaning.py pyspark --deploy-mode client`
- What this does:
    - input:  team3/data_ingest/business.csv
    - output: team3/data_ingest/business_cleaned.csv
    - data profiling nans:
        - output the percentage of rows with any nan values of the original dataset (it’s actually 100%, meaning each roles have at least 1 nan)
        - output number of nans in each column
        - as I dropped useless columns for analysis, output nan percentage of new dataframe again (18%)
    - seperates the coordinate column into columns of longitude and lattitude

## ETL2- Join three datasets, Clean null values

- Join police_short.csv with listings.csv and move the outputs to project dir (NOTE: The listings.csv file here is my own hdfs)
`module load python/gcc/3.7.9
PYTHONSTARTUP=project/cleaning3.py pyspark --deploy-mode client
mv Airbnb_Review_Score_PCA.png project
mv crime_number vs price.png project`
    - output: `airbnb_police.csv`
- Then join `airbnb_police.csv` and `business_cleaned.csv` together
    - run: rm `final_join.csv` if it already exists
    - run:
        
        `module load python/gcc/3.7.
        PYTHONSTARTUP=final_join.py pyspark --deploy-mode client`
        
    - what this script does:
        - join the 3 datasets into 1 final dataset
        - data profile: count the nan values and drop the rows with nan values
    - output: `team3/final_join.csv`
    - `team3/final_join.csv` is the final joined dataset that we will perform analysis on
        - it is devoid of any nan values

## Data Profiling (Post-Joint) - Inspect/Clean data, Calculate descriptive statistics on the joint data

### Profiling on police_short

- upload the police_short_profile.py to peel and hdfs
    
    `scp /Users/caoming/malavet/project/police_short_profile.py mc7787@peel.hpc.nyu.edu:/home/mc7787/team3/profiling_code`
    
    `hdfs dfs -put team3/profiling_code/police_short_profile.py team3/profiling_code`
    `module load python/gcc/3.7.9`
    
    `PYTHONSTARTUP=team3/profiling_code/police_short_profile.py pyspark --deploy-mode client`
    

### Profiling for airbnb and business data are included in the cleaning process

### Profiling on joint data

- run `module load python/gcc/3.7.9`
- run `PYTHONSTARTUP=descriptive_stats.py pyspark --deploy-mode client`
- some other data profiling is also done in `business_cleaning.py` and `final_join.py`

## Data Analysis - Spearman Correlation, Linear Regression, PCA with PySpark, Spark MLlib

### PCA random forest analysis:

- upload the acode.py to home and hdfs
    - `scp /Users/caoming/malavet/project/acode.py mc7787@peel.hpc.nyu.edu:/home/mc7787/team3/ana_code`
    - hdfs dfs -put team3/ana_code/acode.py team3/ana_code
- `module load python/gcc/3.7.9`
- `PYTHONSTARTUP=team3/ana_code/acode.py pyspark --deploy-mode client`

### Analysis while grouping data by borough:

- run `module load python/gcc/3.7.9`
- run `PYTHONSTARTUP=correlation_by_borough.py pyspark --deploy-mode client`