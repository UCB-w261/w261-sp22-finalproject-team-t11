# Databricks notebook source
from pyspark.sql.functions import col, isnan, when, count
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pyspark
import datetime as dt
import pandas as pd

# Setup Blob store access
blob_container = "w261team11" # The name of your container created in https://portal.azure.com
storage_account = "w261sa" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261team11" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261team11key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# Define constants
# Location of raw data.
IATA_TZ_MAP_RAW_LOC = f"{blob_url}/raw/iata_tz_map"
AIRPORTS_CODE_RAW_LOC = f"{blob_url}/raw/airport_codes"
AIRPORTS_MASTER_RAW_LOC = f"{blob_url}/raw/airports"

# Original given sources:
FLIGHT_RAW_LOC =  "/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*"
WEATHER_STATIONS_RAW_LOC = "/mnt/mids-w261/datasets_final_project/stations_data/*"
WEATHER_RAW_LOC = "/mnt/mids-w261/datasets_final_project/weather_data/*"

# New data sources:
FLIGHT_COVID_RAW_LOC = f"{blob_url}/raw/flights_covid"
# WEATHER_STATIONS_RAW_LOC = f"{blob_url}/raw/stations"
WEATHER_RAW_LOC = f"{blob_url}/raw/weather"

# Location of staged data.
# AIRPORT_WEATHER_LOC = f"{blob_url}/raw/airport_weather"
AIRPORTS_MASTER_LOC = f"{blob_url}/staged/airports"
AIRPORTS_WS_LOC = f"{blob_url}/staged/airports_weatherstations"
WEATHER_LOC = f"{blob_url}/staged/weather"
CLEAN_WEATHER_LOC = f'{WEATHER_LOC}/clean_weather_data.parquet'

# Location of final joined data.
FINAL_JOINED_DATA_ALL = f"{blob_url}/staged/final_joined_all"

SHAPES_BASE_FOLDER = "/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/shapes"

# COMMAND ----------

# MAGIC %md
# MAGIC #Load Data

# COMMAND ----------

# Load the joined data
df = spark.read.parquet(FINAL_JOINED_DATA_ALL)

display(df)

# COMMAND ----------

# count rows and columns in df
print((df.count(), len(df.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC # Drop Data

# COMMAND ----------

def data_filtering(df):
    # Filter out canceled flights
    df = df.where(df['CANCELLED'] != 1)
    
    return df
    
df = data_filtering(df)

# COMMAND ----------

# count rows and columns in df
print((df.count(), len(df.columns)))

# COMMAND ----------

def na_cols(df):
    # FEATURES TO DEFINITELY DROP BECAUSE OF NAs
    features1 = ['DIV4_AIRPORT', 'DIV4_AIRPORT_ID', 'DIV4_AIRPORT_SEQ_ID', 'DIV4_WHEELS_ON', 'DIV4_TOTAL_GTIME', 'DIV4_LONGEST_GTIME'
                'DIV4_WHEELS_OFF', 'DIV4_TAIL_NUM', 'DIV5_AIRPORT', 'DIV5_AIRPORT_ID', 'DIV5_AIRPORT_SEQ_ID', 'DIV5_WHEELS_ON', 
                'DIV5_TOTAL_GTIME', 'DIV5_LONGEST_GTIME', 'DIV5_WHEELS_OFF', 'DIV5_TAIL_NUM', 'DIV3_WHEELS_OFF', 'DIV3_TAIL_NUM', 'DIV3_AIRPORT',
                'DIV3_AIRPORT_ID', 'DIV3_AIRPORT_SEQ_ID', 'DIV3_WHEELS_ON', 'DIV3_TOTAL_GTIME', 'DIV3_LONGEST_GTIME', 'DIV2_WHEELS_OFF',
                'DIV2_TAIL_NUM', 'DIV2_AIRPORT', 'DIV2_WHEELS_ON', 'DIV2_TOTAL_GTIME', 'DIV2_LONGEST_GTIME', 'DIV2_AIRPORT_ID', 
                'DIV2_AIRPORT_SEQ_ID', 'DIV_ACTUAL_ELAPSED_TIME', 'DIV_ARR_DELAY', 'DIV1_WHEELS_OFF', 'DIV1_TAIL_NUM', 'DIV_DISTANCE', 
                'DIV_REACHED_DEST', 'DIV1_AIRPORT', 'DIV1_AIRPORT_ID', 'DIV1_AIRPORT_SEQ_ID', 'DIV1_WHEELS_ON', 'DIV1_TOTAL_GTIME', 
                'DIV1_LONGEST_GTIME', 'TOTAL_ADD_GTIME', 'LONGEST_ADD_GTIME', 'FIRST_DEP_TIME', 'CANCELLATION_CODE', 
                'dest_weather_Avg_HourlyWindGustSpeed', 'origin_weather_Avg_HourlyWindGustSpeed']
    
    # FEATURES TO PROBABLY DROP BECAUSE OF NAs
    features2 = ['origin_weather_Avg_HourlyPressureChange', 'dest_weather_Avg_HourlyPressureChange']
    
    df = df.drop(*features1)
    df = df.drop(*features2)
    
    return df
    
df = na_cols(df) 

# COMMAND ----------

# count rows and columns in df
print((df.count(), len(df.columns)))

# COMMAND ----------

def drop_cols_after_self_referenced_join(df):
    # Get characteristics from the previous flight and then drop the column for the flight you're predicting for
    
    # we don't know delay causes of flight we're predicting the delay for
    delay_causes = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
    
    # we don't know characteristics of arrival delay for the flight we're predicting for
    arrival_delay = ['ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP']
    
    # we don't know how long the current flight will actually last, we should only keep the CRS time
    flight_features = ['AIR_TIME', 'ACTUAL_ELAPSED_TIME', 'WHEELS_ON' 'TAXI_IN', 'ARR_TIME', 'TAXI_OUT', 'WHEELS_OFF']
    
    # we don't know about the current flight delay since this is what we're predicting
    departure_delay = ['DEP_TIME']
    # keep DEP_DEL15, DEP_DELAY, DEP_DELAY_NEW, DEP_DELAY_GROUP since these are features we might use for prediction 
    
    # we don't know if the flight we're predicting for will be diverted
    diversions = ['DIV_AIRPORT_LANDINGS', 'DIVERTED']
    
    # we don't need the exact timestamp once all joins are done
    time_cols = ['_utc_dept_ts', '_utc_dept_minus2_ts', '_dep_time_str', '_local_dept_ts', '_local_dept_minus2_ts',
                'origin_airport_tz', 'dest_airport_tz', 'origin_weather_Datehour', 'dest_weather_Datehour']
        
    
    df = df.drop(*delay_causes)
    df = df.drop(*arrival_delay)
    df = df.drop(*flight_features)
    df = df.drop(*departure_delay)
    df = df.drop(*diversions)
    df = df.drop(*time_cols)
    
    return df
  
df = drop_cols_after_self_referenced_join(df)

# COMMAND ----------

# count rows and columns in df
print((df.count(), len(df.columns)))

# COMMAND ----------

def other_drop_cols(df):
    #other_cols = ['TAIL_NUM', 'FL_DATE', 'DAY_OF_MONTH', 'OP_CARRIER_FL_NUM', 'ORIGIN_WAC', 'DEST_WAC',
                 #'CRS_DEP_TIME', 'CRS_ARR_TIME', 'FLIGHTS']
    other_cols = ['TAIL_NUM', 'DAY_OF_MONTH', 'OP_CARRIER_FL_NUM', 'ORIGIN_WAC', 'DEST_WAC',
                 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'FLIGHTS']
        
    # tail num should be dropped and replaced with features about each aircraft @DANTE we need this data
    # flight date has too many unique values to one-hot encode
    # day of month will likely be irrelevant for a model, we have no reason to believe there should be a relation between this feature and delays
    # flight number has too many unique values to one-hot encode
    # ORIGIN_WAC & DEST_WAC is constant for US flights -- do we care about international flights? Decision to make.
    # CRS_DEP_TIME & CRS_ARR_TIME -- use BLK times instead, we probably don't need the exact time
    # not sure what FLIGHTS even is
    
    unsure_about = ['origin_airport_iso_country', 'origin_airport_iso_region', 'dest_airport_iso_country', 'dest_airport_iso_country']
    
    weather = ['origin_airport_ws_station_id', 'dest_airport_ws_station_id', 'origin_weather_Station', 'origin_weather_Avg_Elevation', 
               'origin_weather_HourlyPressureTendency_Decreasing', 'origin_weather_HourlyPressureTendency_Constant', 'dest_weather_Station', 
               'dest_weather_Avg_Elevation', 'dest_weather_HourlyPressureTendency_Decreasing', 'dest_weather_HourlyPressureTendency_Constant']
    # too many weather stations to one-hot encode, would lead to overfitting
    # drop weather station elevation, use airport elevation instead
    # drop Pressure Tendency Decreasing and Constant features since these are always 0
    df = df.drop(*other_cols)
    df = df.drop(*unsure_about)
    df = df.drop(*weather)

    return df

df = other_drop_cols(df)

# COMMAND ----------

# count rows and columns in df
print((df.count(), len(df.columns)))

# COMMAND ----------

display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

large_col = ['QUARTER',
 'MONTH',
 'DAY_OF_WEEK',
 'FL_DATE',
 'OP_UNIQUE_CARRIER',
 'OP_CARRIER_AIRLINE_ID',
 'OP_CARRIER',
 'ORIGIN_AIRPORT_ID',
 'ORIGIN_AIRPORT_SEQ_ID',
 'ORIGIN_CITY_MARKET_ID',
 'ORIGIN',
 'ORIGIN_CITY_NAME',
 'ORIGIN_STATE_ABR',
 'ORIGIN_STATE_FIPS',
 'ORIGIN_STATE_NM',
 'DEST_AIRPORT_ID',
 'DEST_AIRPORT_SEQ_ID',
 'DEST_CITY_MARKET_ID',
 'DEST',
 'DEST_CITY_NAME',
 'DEST_STATE_ABR',
 'DEST_STATE_FIPS',
 'DEST_STATE_NM',
 'WHEELS_ON',
 'TAXI_IN',
 'ARR_TIME_BLK',
 'CRS_ELAPSED_TIME',
 'DISTANCE',
 'DISTANCE_GROUP',
 'origin_airport_iata',
 'origin_airport_type',
 'origin_airport_elevation',
 'dest_airport_iata',
 'dest_airport_type',
 'dest_airport_elevation',
 'dest_airport_iso_region',
 'origin_weather_Avg_HourlyAltimeterSetting',
 'origin_weather_Avg_HourlyDewPointTemperature',
 'origin_weather_Avg_HourlyDryBulbTemperature',
 'origin_weather_Avg_HourlyRelativeHumidity',
 'origin_weather_Avg_HourlySeaLevelPressure',
 'origin_weather_Avg_HourlyStationPressure',
 'origin_weather_Avg_HourlyVisibility',
 'origin_weather_Avg_HourlyWetBulbTemperature',
 'origin_weather_Avg_HourlyWindDirection',
 'origin_weather_Avg_HourlyWindSpeed',
 'origin_weather_Avg_Precip_Double',
 'origin_weather_Trace_Rain',
 'origin_weather_NonZero_Rain',
 'origin_weather_HourlyPressureTendency_Increasing',
 'origin_weather_Calm_Winds',
 'origin_weather_Sky_Conditions_CLR',
 'origin_weather_Sky_Conditions_FEW',
 'origin_weather_Sky_Conditions_SCT',
 'origin_weather_Sky_Conditions_BKN',
 'origin_weather_Sky_Conditions_OVC',
 'origin_weather_Sky_Conditions_VV',
 'origin_weather_Present_Weather_Drizzle',
 'origin_weather_Present_Weather_Rain',
 'origin_weather_Present_Weather_Snow',
 'origin_weather_Present_Weather_SnowGrains',
 'origin_weather_Present_Weather_IceCrystals',
 'origin_weather_Present_Weather_Hail',
 'origin_weather_Present_Weather_Mist',
 'origin_weather_Present_Weather_Fog',
 'origin_weather_Present_Weather_Smoke',
 'origin_weather_Present_Weather_Dust',
 'origin_weather_Present_Weather_Haze',
 'origin_weather_Present_Weather_Storm',
 'dest_weather_Avg_HourlyAltimeterSetting',
 'dest_weather_Avg_HourlyDewPointTemperature',
 'dest_weather_Avg_HourlyDryBulbTemperature',
 'dest_weather_Avg_HourlyRelativeHumidity',
 'dest_weather_Avg_HourlySeaLevelPressure',
 'dest_weather_Avg_HourlyStationPressure',
 'dest_weather_Avg_HourlyVisibility',
 'dest_weather_Avg_HourlyWetBulbTemperature',
 'dest_weather_Avg_HourlyWindDirection',
 'dest_weather_Avg_HourlyWindSpeed',
 'dest_weather_Avg_Precip_Double',
 'dest_weather_Trace_Rain',
 'dest_weather_NonZero_Rain',
 'dest_weather_HourlyPressureTendency_Increasing',
 'dest_weather_Calm_Winds',
 'dest_weather_Sky_Conditions_CLR',
 'dest_weather_Sky_Conditions_FEW',
 'dest_weather_Sky_Conditions_SCT',
 'dest_weather_Sky_Conditions_BKN',
 'dest_weather_Sky_Conditions_OVC',
 'dest_weather_Sky_Conditions_VV',
 'dest_weather_Present_Weather_Drizzle',
 'dest_weather_Present_Weather_Rain',
 'dest_weather_Present_Weather_Snow',
 'dest_weather_Present_Weather_SnowGrains',
 'dest_weather_Present_Weather_IceCrystals',
 'dest_weather_Present_Weather_Hail',
 'dest_weather_Present_Weather_Mist',
 'dest_weather_Present_Weather_Fog',
 'dest_weather_Present_Weather_Smoke',
 'dest_weather_Present_Weather_Dust',
 'dest_weather_Present_Weather_Haze',
 'dest_weather_Present_Weather_Storm',
 'YEAR',
 'DEP_DEL15']

small_columns = ['QUARTER',
 'MONTH',
 'DAY_OF_WEEK',
 'FL_DATE',
 'origin_weather_Avg_HourlyAltimeterSetting',
 'origin_weather_Avg_HourlyDewPointTemperature',
 'origin_weather_Avg_HourlyDryBulbTemperature',
 'origin_weather_Avg_HourlyRelativeHumidity',
 'origin_weather_Avg_HourlySeaLevelPressure',
 'origin_weather_Avg_HourlyStationPressure',
 'origin_weather_Avg_HourlyVisibility',
 'origin_weather_Avg_HourlyWetBulbTemperature',
 'origin_weather_Avg_HourlyWindDirection',
 'origin_weather_Avg_HourlyWindSpeed',
 'origin_weather_Avg_Precip_Double',
 'origin_weather_Trace_Rain',
 'origin_weather_NonZero_Rain',
 'origin_weather_HourlyPressureTendency_Increasing',
 'dest_weather_Avg_HourlyAltimeterSetting',
 'dest_weather_Avg_HourlyDewPointTemperature',
 'dest_weather_Avg_HourlyDryBulbTemperature',
 'dest_weather_Avg_HourlyRelativeHumidity',
 'dest_weather_Avg_HourlySeaLevelPressure',
 'dest_weather_Avg_HourlyStationPressure',
 'dest_weather_Avg_HourlyVisibility',
 'dest_weather_Avg_HourlyWetBulbTemperature',
 'dest_weather_Avg_HourlyWindDirection',
 'dest_weather_Avg_HourlyWindSpeed',
 'dest_weather_Avg_Precip_Double',
 'dest_weather_Trace_Rain',
 'dest_weather_NonZero_Rain',
 'dest_weather_HourlyPressureTendency_Increasing',
 'YEAR',
 'DEP_DEL15']

df = df.select(small_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC # Split Data 

# COMMAND ----------

from pyspark.sql.functions import to_date

# COMMAND ----------

#df = df.select(col("FL_DATE"),to_date(col("FL_DATE"),"yyyy-mm-dd"))

df = df.withColumn("FL_DATE", to_date("FL_DATE", "yyyy-MM-dd"))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df_train = df.where(df["FL_DATE"] <= '2015-05-31')
#df_test = df.where((df["FL_DATE"] >= '2015-02-28') & (df["FL_DATE"] <= '2015-03-31'))
df_test = df.where((df["FL_DATE"] >= '2015-06-01') & (df["FL_DATE"] <= '2015-07-30'))

# COMMAND ----------

# count rows and columns in df
print((df_train.count(), len(df_train.columns)))

# COMMAND ----------

# count rows and columns in df
print((df_test.count(), len(df_test.columns)))

# COMMAND ----------

df_train = df_train.drop("FL_DATE")
df_test = df_test.drop("FL_DATE")

# COMMAND ----------

# MAGIC %md
# MAGIC # Convert to RDD and Parse

# COMMAND ----------

trainRDD = df_train.rdd
testRDD = df_test.rdd

# COMMAND ----------

def parse(line):
    features,result = line[:-1], line[-1]
    return(features, result)

# COMMAND ----------

trainRDDCached = trainRDD.map(parse).cache()
testRDDCached = testRDD.map(parse).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC # OLS Model

# COMMAND ----------

meanResult = None # FILL IN YOUR CODE HERE
meanResult = trainRDDCached.map(lambda x : x[1]).mean()
varResult = None # FILL IN YOUR CODE HERE
varResult = trainRDDCached.map(lambda x : x[1]).variance()
print(f"Mean: {meanResult}")
print(f"Variance: {varResult}")

# COMMAND ----------

def normalize(dataRDD):
    """
    Scale and center data round mean of each feature.
    Args:
        dataRDD - records are tuples of (features_array, y)
    Returns:
        normedRDD - records are tuples of (features_array, y)
    """
    featureMeans = dataRDD.map(lambda x: x[0]).mean()
    featureStdev = np.sqrt(dataRDD.map(lambda x: x[0]).variance())
    
    ################ YOUR CODE HERE #############
    normedRDD = None
    
    normedRDD = dataRDD.map(lambda x : ((x[0]-featureMeans)/featureStdev,x[1]))
    #(np.append([1.0], x[0]), x[1])
    
    ################ FILL IN YOUR CODE HERE #############
    
    return normedRDD

# COMMAND ----------

normedRDD = normalize(trainRDDCached).cache()

# COMMAND ----------

def OLSLoss(dataRDD, W):
    """
    Compute mean squared error.
    Args:
        dataRDD - each record is a tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    """
    print(W)
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1]))
    ################## YOUR CODE HERE ##################
    loss = None
    X = np.array(augmentedData.map(lambda x : x[0]).collect())
    y = augmentedData.map(lambda x : x[1]).collect()
    N = len(X)
    loss = 1/float(N) * sum((W.dot(X.T) - y)**2)
    
    ################## (END) YOUR CODE ##################
    return loss


# COMMAND ----------

def OLSGradient(dataRDD, W):
            X = np.array(augmentedData.map(lambda x : x[0]).collect())
            y = augmentedData.map(lambda x : x[1]).collect()
            N = len(X)
            return 2.0/N *(W.dot(X.T) - y).dot(X)

# COMMAND ----------

BASELINE = np.array([meanResult,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
])

# COMMAND ----------

def GDUpdate(dataRDD, W, learningRate = 0.1):
    """
    Perform one OLS gradient descent step/update.
    Args:
        dataRDD - records are tuples of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    Returns:
        new_model - (array) updated coefficients, bias at index 0
    """
    # add a bias 'feature' of 1 at index 0
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1])).cache()
    
    ################## YOUR CODE HERE ################# 
    def OLSGradient(dataRDD, W):
        
        X = np.array(augmentedData.map(lambda x : x[0]).collect())
        y = augmentedData.map(lambda x : x[1]).collect()
        N = len(X)
        return 2.0/N *(W.dot(X.T) - y).dot(X)
    
    grad = None
    new_model = None 
    
   
    loss = OLSLoss(dataRDD,W)
    
    gradient = OLSGradient(augmentedData, W)
        
    update = np.multiply(gradient,learningRate)
        
    new_model = W - update
    
    ################## (END) YOUR CODE ################# 
   
    return new_model
    #return gradient

# COMMAND ----------

nSteps = 5
model = BASELINE
print(f"BASELINE:  Loss = {OLSLoss(trainRDDCached,model)}")
for idx in range(nSteps):
    print("----------")
    print(f"STEP: {idx+1}")
    model = GDUpdate(trainRDDCached, model)
    loss = OLSLoss(trainRDDCached, model)
    print(f"Loss: {loss}")
    print(f"Model: {[round(w,3) for w in model]}")

# COMMAND ----------

def GradientDescent(trainRDD, testRDD, wInit, nSteps = 20, 
                    learningRate = 0.1, verbose = False):
    """
    Perform nSteps iterations of OLS gradient descent and 
    track loss on a test and train set. Return lists of
    test/train loss and the models themselves.
    """
    # initialize lists to track model performance
    train_history, test_history, model_history = [], [], []
    
    # add a bias 'feature' of 1 at index 0
    augmentedtrain = trainRDD.map(lambda x: (np.append([1.0], x[0]), x[1])).cache()
    
    # perform n updates & compute test and train loss after each
    model = wInit
    for idx in range(nSteps): 
        
        ############## YOUR CODE HERE #############
        #model = None
        training_loss = None
        test_loss = None
        
        training_loss = OLSLoss(trainRDD,model)
        
        test_loss = OLSLoss(testRDD,model)
    
        model = GDUpdate(trainRDD, model)
        
        ############## (END) YOUR CODE #############
        
        # keep track of test/train loss for plotting
        train_history.append(training_loss)
        test_history.append(test_loss)
        model_history.append(model)
        
        # console output if desired
        if verbose:
            print("----------")
            print(f"STEP: {idx+1}")
            print(f"training loss: {training_loss}")
            print(f"test loss: {test_loss}")
            print(f"Model: {[round(w,3) for w in model]}")
    return train_history, test_history, model_history
