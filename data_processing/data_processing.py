# Databricks notebook source
# MAGIC %md
# MAGIC # Data Processing

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library Imports & Setup

# COMMAND ----------

from pyspark.sql.functions import col, isnan, when, count, countDistinct, datediff, max, hour, substring, lead, current_timestamp
from pyspark.sql.types import StructType, StructField, TimestampType, StringType, IntegerType, FloatType

from pyspark.ml.functions import vector_to_array
from pyspark.sql import Window
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StringIndexerModel, OneHotEncoderModel, StandardScaler, StandardScalerModel, VectorAssembler, MinMaxScaler, MinMaxScalerModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pyspark
import datetime as dt
from datetime import date, timedelta
import pandas as pd
import holidays
import datetime

from configuration_v01 import Configuration
configuration = Configuration()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Remove Cancelled Flights

# COMMAND ----------

def data_filtering(df):
    # filter out cancelled flights
    df = df.where(df['CANCELLED'] != 1)
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Self-Referenced Join

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Self-join strategy: Find the previous flight with a departure time at least 2 hours before flight departure, but no more than 24 hours earlier than the flight. Use tail numbers to track the fact that two flights are operated using the same aircraft.  We then determine if we have enough information to know 2 hours before departure that the incoming flight might be delayed (i.e. it has been delayed on departure).
# MAGIC 
# MAGIC Our intuition and the EDA on the 

# COMMAND ----------

def self_reference_join(df):
    windowSpec = Window.partitionBy('TAIL_NUM').orderBy(col('_utc_dept_ts').desc())

    df_join = df.withColumn('prior_delayed_all', lead('DEP_DEL15', 1).over(windowSpec)) \
                .withColumn('prior_FL_DATE', lead('FL_DATE', 1).over(windowSpec)) \
                .withColumn('prior_utc_dept_ts', lead('_utc_dept_ts', 1).over(windowSpec)) \
                .withColumn('timediff', col('_utc_dept_ts').cast('long') - col('prior_utc_dept_ts').cast('long')) \
                .withColumn('prior_in', (col('timediff')>8100).cast('integer')) \
                .withColumn('same_day', (col('FL_DATE')==col('prior_FL_DATE')).cast('integer')) \
                .withColumn('prior_delayed', col('prior_delayed_all')*col('prior_in')*col('same_day'))
    df_join = df_join.drop('prior_FL_DATE', 'same_day', 'prior_delayed_all', 'prior_utc_dept_ts', 'timediff', 'prior_in')
    
    return df_join

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drop Columns
# MAGIC - Redundant columns (ex: carrier ID)
# MAGIC - Features with a lot of null values
# MAGIC - Departure performance info (*keep DEP_DEL15 = label)
# MAGIC - Arrival performance info
# MAGIC - Cancellations and Diversion info
# MAGIC - Cause of Delay info
# MAGIC - Gate Return Info at Origin Airport
# MAGIC - Diverted Flight Info

# COMMAND ----------

def na_cols(df):
    # Drop columns with high null values
    features = ['DIV4_AIRPORT', 'DIV4_AIRPORT_ID', 'DIV4_AIRPORT_SEQ_ID', 'DIV4_WHEELS_ON', 'DIV4_TOTAL_GTIME', 'DIV4_LONGEST_GTIME','DIV4_WHEELS_OFF', 'DIV4_TAIL_NUM', 'DIV5_AIRPORT', 'DIV5_AIRPORT_ID', 'DIV5_AIRPORT_SEQ_ID', 'DIV5_WHEELS_ON','DIV5_TOTAL_GTIME', 'DIV5_LONGEST_GTIME', 'DIV5_WHEELS_OFF', 'DIV5_TAIL_NUM', 'DIV3_WHEELS_OFF', 'DIV3_TAIL_NUM', 'DIV3_AIRPORT','DIV3_AIRPORT_ID', 'DIV3_AIRPORT_SEQ_ID', 'DIV3_WHEELS_ON', 'DIV3_TOTAL_GTIME', 'DIV3_LONGEST_GTIME', 'DIV2_WHEELS_OFF','DIV2_TAIL_NUM', 'DIV2_AIRPORT', 'DIV2_WHEELS_ON', 'DIV2_TOTAL_GTIME', 'DIV2_LONGEST_GTIME', 'DIV2_AIRPORT_ID','DIV2_AIRPORT_SEQ_ID', 'DIV_ACTUAL_ELAPSED_TIME', 'DIV_ARR_DELAY', 'DIV1_WHEELS_OFF', 'DIV1_TAIL_NUM', 'DIV_DISTANCE','DIV_REACHED_DEST', 'DIV1_AIRPORT', 'DIV1_AIRPORT_ID', 'DIV1_AIRPORT_SEQ_ID', 'DIV1_WHEELS_ON', 'DIV1_TOTAL_GTIME','DIV1_LONGEST_GTIME', 'TOTAL_ADD_GTIME', 'LONGEST_ADD_GTIME', 'FIRST_DEP_TIME', 'CANCELLATION_CODE', 'dest_weather_Avg_HourlyWindGustSpeed', 'origin_weather_Avg_HourlyWindGustSpeed','origin_weather_Avg_HourlyPressureChange', 'dest_weather_Avg_HourlyPressureChange']
    
    df = df.drop(*features)
    return df

# COMMAND ----------

def drop_cols_after_self_referenced_join(df):
    # Get characteristics from the previous flight and then drop the column for the flight you're predicting for
    
    # we don't know delay causes of flight we're predicting the delay for
    delay_causes = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
    
    # we don't know characteristics of arrival delay for the flight we're predicting for
    arrival_delay = ['ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP']
    
    # we don't know how long the current flight will actually last, we should only keep the CRS time
    flight_features = ['AIR_TIME', 'ACTUAL_ELAPSED_TIME', 'WHEELS_ON', 'TAXI_IN', 'ARR_TIME', 'TAXI_OUT', 'WHEELS_OFF', 
                       'DEP_TIME_BLK', 'ARR_TIME_BLK']
    
    # we don't know about the current flight delay since this is what we're predicting
    departure_delay = ['DEP_TIME']
    # keep DEP_DEL15, DEP_DELAY, DEP_DELAY_NEW, DEP_DELAY_GROUP since these are features we might use for prediction 
    
    # we don't know if the flight we're predicting for will be diverted
    diversions = ['DIV_AIRPORT_LANDINGS', 'DIVERTED']
    
    # we don't need the exact timestamp once all joins are done
    time_cols = ['_utc_dept_minus2_ts', '_local_dept_minus2_ts',
                'origin_airport_tz', 'dest_airport_tz', 'origin_weather_Datehour', 'dest_weather_Datehour']
    
    # drop duplicate calls that resulted from self-referenced join
    duplicate_cols = ['flight_dept_utc', 'tail_num']
    
    df = df.drop(*delay_causes).drop(*arrival_delay).drop(*flight_features).drop(*departure_delay).drop(*diversions)\
        .drop(*time_cols).drop(*duplicate_cols)
    return df

# COMMAND ----------

# dummy function
def exclude_features_from_model():
    """This function defines variables that are kept in the dataset but should be excluded from the model."""
    
    # make sure these features are not used when training our model (potential labels in model)
    # use YEAR only for paritioning data
    exclude_features = ['DEP_DEL15', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DELAY_GROUP', 'YEAR']
    
    # add timestamps here instead of dropping them entirely -- helps us track outliers
    # flight identifiers: time (scheduled and actual), origin airport, departure airport, tail number, flight number
    reference_cols = ['_utc_dept_ts', '_local_dept_ts', '_local_dept_actual_ts', 'local_at_src_airport_arr_ts',
                      '_local_at_src_airport_arr_actual_ts', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'TAIL_NUM', 'OP_CARRIER_FL_NUM']

# COMMAND ----------

def other_drop_cols(df):
    other_cols = ['DAY_OF_MONTH', 'ORIGIN_WAC', 'DEST_WAC',
                 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'FLIGHTS']
    # day of month will likely be irrelevant for a model, we have no reason to believe there should be a relation between this feature and delays
    # ORIGIN_WAC & DEST_WAC is constant for US flights
    # CRS_DEP_TIME & CRS_ARR_TIME -- use BLK times instead, we probably don't need the exact time
    # not sure what FLIGHTS even is, always equal to 1
    
    # keep tail number for reference, but we're not one-hot encoding this
    # keep flight number for reference, but we're not one-hot encoding this
    
    unsure_about = ['origin_airport_iso_country', 'origin_airport_iso_region', 'dest_airport_iso_country', 'dest_airport_iso_region']
    # origin_airport_iso_country has 3 values: VI, US, PR
    
    weather = ['origin_airport_ws_station_id', 'dest_airport_ws_station_id', 'origin_weather_Station',
               'origin_weather_HourlyPressureTendency_Decreasing', 'origin_weather_HourlyPressureTendency_Constant', 'dest_weather_Station', 
               'dest_weather_HourlyPressureTendency_Decreasing', 'dest_weather_HourlyPressureTendency_Constant']
    # too many weather stations to one-hot encode, would lead to overfitting
    # keep weather station elevation in case there is not a weather station at the airport -- 
       # 'origin_weather_Avg_Elevation', 'dest_weather_Avg_Elevation'
    # drop Pressure Tendency Decreasing and Constant features since these are always 0
    
    ### DROP dupilicate categorical Features; keeping OP_UNIQUE_CARRIER
    airline = ['OP_CARRIER_AIRLINE_ID', 'OP_CARRIER']
    
    # ORIGIN FEATURES
    # keep 'ORIGIN_AIRPORT_ID' for reference purposes
    orig_airport = ['ORIGIN_AIRPORT_SEQ_ID', 'origin_airport_iata']
    orig_city = ['ORIGIN_CITY_NAME', 'ORIGIN_CITY_MARKET_ID']
    orig_state = ['ORIGIN_STATE_ABR', 'ORIGIN_STATE_FIPS', 'ORIGIN_STATE_NM']
    orig_all = orig_airport + orig_city + orig_state
    
    # DESTINATION FEATURES
    # keep 'DEST_AIRPORT_ID' for reference purposes
    dest_airport = ['DEST_AIRPORT_SEQ_ID', 'dest_airport_iata']
    dest_city = ['DEST_CITY_NAME', 'DEST_CITY_MARKET_ID']
    dest_state = ['DEST_STATE_FIPS', 'DEST_STATE_NM', 'DEST_STATE_ABR']
    dest_all = dest_airport + dest_city + dest_state
    
    df = df.drop(*other_cols).drop(*unsure_about).drop(*weather).drop(*airline).drop(*orig_all).drop(*dest_all)
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Address Null Values

# COMMAND ----------

# MAGIC %md
# MAGIC - investigate `DEP_DEL15` nulls
# MAGIC - address nulls in weather data

# COMMAND ----------

def drop_null_rows(df):
    # DEP_DEL15 --> flights with nulls have no actual departure time or actual arrival time. 
    # They appear to be canceled even though `CANCELED` is marked as 0
    df = df.na.drop(subset = ['DEP_DEL15', 'origin_weather_Avg_HourlyWindDirection', 'dest_weather_Avg_HourlyWindDirection', 
                         'dest_weather_Avg_HourlySeaLevelPressure', 'origin_weather_Avg_HourlySeaLevelPressure', 
                         'origin_weather_Avg_HourlyWetBulbTemperature', 'dest_weather_Avg_HourlyWetBulbTemperature', 
                         'origin_weather_Avg_HourlyStationPressure', 'dest_weather_Avg_HourlyStationPressure', 
                         'dest_weather_Avg_HourlyRelativeHumidity', 'dest_weather_Avg_HourlyDewPointTemperature', 
                         'origin_weather_Avg_HourlyRelativeHumidity', 'origin_weather_Avg_HourlyDewPointTemperature', 
                         'dest_weather_Avg_HourlyWindSpeed', 'origin_weather_Avg_HourlyWindSpeed', 'dest_weather_Avg_HourlyDryBulbTemperature', 
                         'origin_weather_Avg_HourlyDryBulbTemperature', 'dest_weather_Avg_HourlyVisibility', 
                         'origin_weather_Avg_HourlyVisibility', 'origin_weather_Avg_HourlyAltimeterSetting', 
                         'dest_weather_Avg_HourlyAltimeterSetting', 'CRS_ELAPSED_TIME'])
    
    
    # drop in future if we want to explore these columns
    future_drops = ['DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DELAY_GROUP']
    
    return df


# COMMAND ----------

def address_prior_delayed_nulls(df):
    # fill in NAs in `prior_delayed` with 0
    df = df.fillna(value = 0, subset = ['prior_delayed'])
    
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## One-Hot Encoding
# MAGIC - Convert categorical variables (string columns) to numeric values using StringIndexer, then apply OneHotEncoder 
# MAGIC - see here for example: https://medium.com/@nutanbhogendrasharma/role-of-onehotencoder-and-pipelines-in-pyspark-ml-feature-part-2-3275767e74f0

# COMMAND ----------

def one_hot_encode_features(df, train_data, SI_MODEL_LOC, OHE_MODEL_LOC):
    # the time_cols are already integers, no need to use StringIndexer
    df = df.withColumn('CRS_DEPT_HR', substring(col('_dep_time_str'), 1, 2).cast("integer"))

    time_cols = ['QUARTER', 'MONTH', 'DAY_OF_WEEK', 'CRS_DEPT_HR', 'DISTANCE_GROUP']
    time_cols_OHE = ['_QUARTER', '_MONTH', '_DAY_OF_WEEK', '_CRS_DEPT_HR', '_DISTANCE_GROUP']
    time_cols_drop = ['DAY_OF_WEEK', 'CRS_DEPT_HR', 'DISTANCE_GROUP']
    
    # airline feature = 'OP_UNIQUE_CARRIER'
    infrequent_delays = ['HA', 'AS', 'DL', 'US', 'YX', 'OO'] # airlines with delay frequencies < 17%
    midlevel_delays = ['9E', 'AA', 'EV', 'MQ', 'YV', 'UA', 'OH'] # airlines with midlevels of delay frequencies >= 17%, < 20%
    frequent_delays = ['G4', 'NK', 'WN', 'VX', 'F9', 'B6']
    # categorize airline according to their delay frequency. if an airline is not in the list, default it as an airline with midlevel delays
    df = df.withColumn('AIRLINE_DELAYS', when(df['OP_UNIQUE_CARRIER'].isin(infrequent_delays), "infrequent")\
                       .when(df['OP_UNIQUE_CARRIER'].isin(frequent_delays), "frequent")\
                       .otherwise("middle"))
    airline = ['AIRLINE_DELAYS'] # instead of one-hot encoding the airline, group by delay frequency
    airline_SI = ['AIRLINE_DELAYS_SI']
    airline_OHE = ['_AIRLINE_DELAYS']

    # ORIGIN FEATURES
    orig_airport = ['origin_airport_type']
    orig_airport_SI = ['origin_airport_type_SI']
    orig_airport_OHE = ['_origin_airport_type']

    # DESTINATION FEATURES
    dest_airport = ['dest_airport_type']
    dest_airport_SI = ['dest_airport_type_SI']
    dest_airport_OHE = ['_dest_airport_type']
    
    cols_to_string_indexer = airline + orig_airport + dest_airport
    new_string_indexed_cols = airline_SI + orig_airport_SI + dest_airport_SI
    all_cols_to_encode = time_cols + airline_SI + orig_airport_SI + dest_airport_SI
    all_cols_to_encode_drop = time_cols_drop + airline_SI + orig_airport_SI + dest_airport_SI
    new_encoded_cols = time_cols_OHE + airline_OHE + orig_airport_OHE + dest_airport_OHE
    
    feature_indexer = StringIndexer(inputCols = cols_to_string_indexer, outputCols = new_string_indexed_cols)
    if train_data:
        index_model = feature_indexer.fit(df)
        index_model.write().overwrite().save(SI_MODEL_LOC)
    else:
        index_model = StringIndexerModel.load(SI_MODEL_LOC)
    df_indexed = index_model.transform(df)
    
    encoder = OneHotEncoder(inputCols = all_cols_to_encode, outputCols = new_encoded_cols)
    if train_data:
        encoder_model = encoder.fit(df_indexed)
        encoder_model.write().overwrite().save(OHE_MODEL_LOC)
    else:
        encoder_model = OneHotEncoderModel.load(OHE_MODEL_LOC)
    df_encoded = encoder_model.transform(df_indexed)
    df_encoded = df_encoded.drop(*cols_to_string_indexer).drop(*new_string_indexed_cols).drop(*all_cols_to_encode_drop)
    
    return df_encoded


# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalization
# MAGIC - Use Vector Assembler and Standard Scaler to center/scale the data 
# MAGIC - We use the train data to normalize the test data

# COMMAND ----------

def normalize(df, train_bool):
    df_cols = set(df.columns)
    
    input_cols = ['origin_weather_Avg_HourlyWindDirection', 'dest_weather_Avg_HourlyWindDirection', 'dest_weather_Avg_HourlySeaLevelPressure', 
                  'origin_weather_Avg_HourlySeaLevelPressure', 'origin_weather_Avg_HourlyWetBulbTemperature', 
                  'dest_weather_Avg_HourlyWetBulbTemperature', 'origin_weather_Avg_HourlyStationPressure', 
                  'dest_weather_Avg_HourlyStationPressure', 'dest_weather_Avg_HourlyRelativeHumidity', 
                  'dest_weather_Avg_HourlyDewPointTemperature', 'origin_weather_Avg_HourlyRelativeHumidity',
                  'origin_weather_Avg_HourlyDewPointTemperature', 'dest_weather_Avg_HourlyWindSpeed',
                  'origin_weather_Avg_HourlyWindSpeed', 'dest_weather_Avg_HourlyDryBulbTemperature',
                  'origin_weather_Avg_HourlyDryBulbTemperature', 'dest_weather_Avg_HourlyVisibility',
                  'origin_weather_Avg_HourlyVisibility', 'origin_weather_Avg_HourlyAltimeterSetting',
                  'dest_weather_Avg_HourlyAltimeterSetting', 'CRS_ELAPSED_TIME', 'DISTANCE',
                  'origin_airport_elevation', 'dest_airport_elevation', 'origin_weather_Avg_Elevation', 'dest_weather_Avg_Elevation']
    output_cols = ['_origin_weather_Avg_HourlyWindDirection', '_dest_weather_Avg_HourlyWindDirection', 
                   '_dest_weather_Avg_HourlySeaLevelPressure', 
                  '_origin_weather_Avg_HourlySeaLevelPressure', '_origin_weather_Avg_HourlyWetBulbTemperature', 
                  '_dest_weather_Avg_HourlyWetBulbTemperature', '_origin_weather_Avg_HourlyStationPressure', 
                  '_dest_weather_Avg_HourlyStationPressure', '_dest_weather_Avg_HourlyRelativeHumidity', 
                  '_dest_weather_Avg_HourlyDewPointTemperature', '_origin_weather_Avg_HourlyRelativeHumidity',
                  '_origin_weather_Avg_HourlyDewPointTemperature', '_dest_weather_Avg_HourlyWindSpeed',
                  '_origin_weather_Avg_HourlyWindSpeed', '_dest_weather_Avg_HourlyDryBulbTemperature',
                  '_origin_weather_Avg_HourlyDryBulbTemperature', '_dest_weather_Avg_HourlyVisibility',
                  '_origin_weather_Avg_HourlyVisibility', '_origin_weather_Avg_HourlyAltimeterSetting',
                  '_dest_weather_Avg_HourlyAltimeterSetting', '_CRS_ELAPSED_TIME', '_DISTANCE',
                  '_origin_airport_elevation', '_dest_airport_elevation', '_origin_weather_Avg_Elevation', '_dest_weather_Avg_Elevation']
    
    non_standardized_features = list(df_cols.difference(set(input_cols)))
    
    vectorized_col = 'vectorized_features'
    
    assembler = VectorAssembler().setInputCols(input_cols).setOutputCol(vectorized_col)
    df_vectorized = assembler.transform(df)
    
#     scaler = StandardScaler(inputCol = vectorized_col, outputCol = "scaled_features")
    scaler = MinMaxScaler(outputCol = "scaled_features").setInputCol(vectorized_col)
    # .setInputCol(vectorized_col)
    # .setOutputCol("scaled_features")
    
    if train_bool:
        scaler_model = scaler.fit(df_vectorized)
        scaler_model.write().overwrite().save(configuration.SCALER_MODEL_LOC)
    else:
        scaler_model = MinMaxScalerModel.load(configuration.SCALER_MODEL_LOC)
    
    df_scaled = scaler_model.transform(df_vectorized)
    
    df_scaled = (df_scaled.withColumn("feat", vector_to_array("scaled_features")))\
                .select(non_standardized_features + [col("feat")[i] for i in range(len(input_cols))])
    
    for i in range(len(output_cols)):
        df_scaled = df_scaled.withColumnRenamed("feat[" + str(i) + "]" , output_cols[i])
    
    df_scaled = df_scaled.drop(*input_cols)
    df_scaled = df_scaled.drop('feat')
    #df_scaled = df_scaled.drop('scaled_features')
    df_scaled = df_scaled.drop('vectorized_features')
    
    return df_scaled

# COMMAND ----------

def normalize_2(df, df_2):
    input_cols = ['origin_weather_Avg_HourlyWindDirection', 'dest_weather_Avg_HourlyWindDirection', 
                  'dest_weather_Avg_HourlySeaLevelPressure', 
                  'origin_weather_Avg_HourlySeaLevelPressure', 'origin_weather_Avg_HourlyWetBulbTemperature', 
                  'dest_weather_Avg_HourlyWetBulbTemperature', 'origin_weather_Avg_HourlyStationPressure', 
                  'dest_weather_Avg_HourlyStationPressure', 'dest_weather_Avg_HourlyRelativeHumidity', 
                  'dest_weather_Avg_HourlyDewPointTemperature', 'origin_weather_Avg_HourlyRelativeHumidity',
                  'origin_weather_Avg_HourlyDewPointTemperature', 'dest_weather_Avg_HourlyWindSpeed',
                  'origin_weather_Avg_HourlyWindSpeed', 'dest_weather_Avg_HourlyDryBulbTemperature',
                  'origin_weather_Avg_HourlyDryBulbTemperature', 'dest_weather_Avg_HourlyVisibility',
                  'origin_weather_Avg_HourlyVisibility', 'origin_weather_Avg_HourlyAltimeterSetting',
                  'dest_weather_Avg_HourlyAltimeterSetting', 'CRS_ELAPSED_TIME', 'DISTANCE',
                  'origin_airport_elevation', 'dest_airport_elevation', 'origin_weather_Avg_Elevation', 'dest_weather_Avg_Elevation']
    output_cols = ['_origin_weather_Avg_HourlyWindDirection', '_dest_weather_Avg_HourlyWindDirection', 
                   '_dest_weather_Avg_HourlySeaLevelPressure', 
                  '_origin_weather_Avg_HourlySeaLevelPressure', '_origin_weather_Avg_HourlyWetBulbTemperature', 
                  '_dest_weather_Avg_HourlyWetBulbTemperature', '_origin_weather_Avg_HourlyStationPressure', 
                  '_dest_weather_Avg_HourlyStationPressure', '_dest_weather_Avg_HourlyRelativeHumidity', 
                  '_dest_weather_Avg_HourlyDewPointTemperature', '_origin_weather_Avg_HourlyRelativeHumidity',
                  '_origin_weather_Avg_HourlyDewPointTemperature', '_dest_weather_Avg_HourlyWindSpeed',
                  '_origin_weather_Avg_HourlyWindSpeed', '_dest_weather_Avg_HourlyDryBulbTemperature',
                  '_origin_weather_Avg_HourlyDryBulbTemperature', '_dest_weather_Avg_HourlyVisibility',
                  '_origin_weather_Avg_HourlyVisibility', '_origin_weather_Avg_HourlyAltimeterSetting',
                  '_dest_weather_Avg_HourlyAltimeterSetting', '_CRS_ELAPSED_TIME', '_DISTANCE',
                  '_origin_airport_elevation', '_dest_airport_elevation', '_origin_weather_Avg_Elevation', '_dest_weather_Avg_Elevation']
    
    w = Window.partitionBy('group')
    
    for i in range(len(input_cols)):
        input_col = input_cols[i]
        output_col = output_cols[i]
        train_mean = mean(df[input_col])
        train_stddev = stddev(df[input_col])
        df = df.withColumn(output_col, (col(input_col) - train_mean)/train_stddev)
        df_2 = df_2.withColumn(output_col, (col(input_col) - train_mean)/train_stddev)
        
    df = df.drop(*input_cols)
    df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

# Create new column that shows average number of flights delayed for each ORIGIN_AIRPORT_ID

def origin_airport_avg_delay(df):
    # calculate average number of flights delayed for each origin airport
    avedelay_by_origin = df.groupBy('ORIGIN_AIRPORT_ID').mean("DEP_DEL15").sort("avg(DEP_DEL15)", ascending=False).toPandas()
    avedelay_by_origin.set_index('ORIGIN_AIRPORT_ID', inplace=True)

    def _helper(x):
        # this helper function looks up the Origin airport and returns the average number of flights delayed for that airport
        y = avedelay_by_origin.loc[x]["avg(DEP_DEL15)"]
        return float(y)

    helperudf = udf(lambda x: _helper(x))

    df = df.withColumn('ORIGIN_AIRPORT_AVG_DELAY', helperudf(col('ORIGIN_AIRPORT_ID')).cast("float"))
    return df 
 


# COMMAND ----------

# create a new column called 'Holiday': (1=yes, 0=no) and 'Holiday_window' if departure date is within 5 days of a holiday: (1=yes, 0=no)
# drop FL_DATE at the end

def add_holidays(df):

    us = holidays.US()

    def _helper(x):
        year, month, day = x.split('-')
        _date = date(year=int(year), month=int(month), day=int(day))
        # returns 1 if x is a US holiday, 0 if not
        return (_date in us)*1

    _helperudf = udf(lambda x: _helper(x))

    df = df.withColumn('Holiday', _helperudf(col('FL_DATE')).cast("integer"))


    # create a new column 'Holiday_window' 
    def _helper(x):
        year, month, day = x.split('-')
        _date = date(year=int(year), month=int(month), day=int(day))
        for delta in range(-5,6):
            if _date + timedelta(days=delta) in us:
                return 1
            return 0

    _helperudf = udf(lambda x: _helper(x))

    df = df.withColumn('Holiday_5Day', _helperudf(col('FL_DATE')).cast("integer"))
    return df

# COMMAND ----------

# create new column for flight departure hour
def add_dep_hour(df):
    
    def _helper(x):
        hour = int(x[:2])
        return hour

    _helperudf = udf(lambda x: _helper(x))

    df = df.withColumn('DEP_HOUR', _helperudf(col('CRS_DEP_TIME')).cast("string"))

    return df

# COMMAND ----------

# Create new column that shows average number of flights delayed for each HOUR

def dep_hour_avg_delay(df):
    # calculate average number of flights delayed for each hour
    avedelay_by_hour = df.groupBy('DEP_HOUR').mean("DEP_DEL15").sort("avg(DEP_DEL15)", ascending=False).toPandas()
    avedelay_by_hour.set_index('DEP_HOUR', inplace=True)

    def _helper(x):
        # this helper function looks up the departure hour and returns the average number of flights delayed for that hour
      y = avedelay_by_hour.loc[x]["avg(DEP_DEL15)"]
      return float(y)

    helperudf = udf(lambda x: _helper(x))

    df = df.withColumn('DEP_HOUR_AVG_DELAY', helperudf(col('DEP_HOUR')))
    return df 

# COMMAND ----------

# MAGIC %md
# MAGIC # Transform and Store

# COMMAND ----------

def file_exists(path):
      try:
        dbutils.fs.ls(path)
        return True
      except Exception as e:
        if 'java.io.FileNotFoundException' in str(e):
            return False
        else:
            raise

def append_audit_record(data_audit, dataset_name, source, destination, sequence, step, rows_after, cols_after):
    ts = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
    data_audit.append([ts, dataset_name, source, destination, sequence, step, rows_after, cols_after])
    
    schema = StructType([
        StructField('timestamp', StringType(), True),
        StructField('dataset_name', StringType(), True),
        StructField('source', StringType(), True),
        StructField('destination', StringType(), True),
        StructField('sequence', IntegerType(), True),
        StructField('STEP', StringType(), True),
        StructField('rows_after', IntegerType(), True),
        StructField('cols_after', IntegerType(), True)
       ])
    return data_audit, sequence + 1, schema

def save_data_audit(data_audit, schema, location, overwrite=False):
    df = spark.createDataFrame(data=data_audit, schema=schema)

    if not overwrite:
        if file_exists(location):
            df_existing = spark.read.parquet(location)
            df = df.union(df_existing)
    
    df.write.mode('overwrite').parquet(location)
    
def tranform_and_store(dataset_name, source, destination, train_bool):
    audit_schema = None
    data_audit = [] 
    sequence = 1

    df = spark.read.parquet(source)
    print("************RAW DATA***************")
    num_rows = df.count()
    num_cols = len(df.columns)
    print(f"Number of rows and columns: {num_rows}, {num_cols}")
    step = 'initial'
    data_audit, sequence, audit_schema = append_audit_record(data_audit, dataset_name, source, destination, sequence, step, num_rows, num_cols)
    
    df_filtered = data_filtering(df)
    num_rows = df_filtered.count()
    num_cols = len(df_filtered.columns)
    print("*****REMOVE CANCELLED FLIGHTS******")
    print(f"Number of rows and columns: {num_rows}, {num_cols}")
    step = 'removed cancelled flights'
    data_audit, sequence, audit_schema = append_audit_record(data_audit, dataset_name, source, destination, sequence, step, num_rows, num_cols)

    df_self_join = self_reference_join(df_filtered) 
    num_rows = df_self_join.count()
    num_cols = len(df_self_join.columns)
    print("*********PRIOR_DELAY ADDED*********")
    print(f"Number of rows and columns: {num_rows}, {num_cols}")
    print(f"Number of delayed incoming flights: {df_self_join.where(df_self_join['prior_delayed'] == 1).count()}")
    step = 'incoming flight delay added'
    data_audit, sequence, audit_schema = append_audit_record(data_audit, dataset_name, source, destination, sequence, step, num_rows, num_cols)

    df_holidays = origin_airport_avg_delay(add_holidays(df_self_join))
    num_rows = df_holidays.count()
    num_cols = len(df_holidays.columns)
    print("*********ADDING HOLIDAYS***********")
    print(f"Number of rows and columns: {num_rows}, {num_cols}")
    step = 'holiday indicator added'
    data_audit, sequence, audit_schema = append_audit_record(data_audit, dataset_name, source, destination, sequence, step, num_rows, num_cols)
    
    df_selected = other_drop_cols(drop_cols_after_self_referenced_join(na_cols(df_holidays)))
    num_rows = df_selected.count()
    num_cols = len(df_selected.columns)
    print("**********SELECTED DATA************")
    print(f"Number of rows and columns: {num_rows}, {num_cols}")
    step = 'dropped useless cols'
    data_audit, sequence, audit_schema = append_audit_record(data_audit, dataset_name, source, destination, sequence, step, num_rows, num_cols)
    
    df_no_nulls = address_prior_delayed_nulls(drop_null_rows(df_selected))
    num_rows = df_no_nulls.count()
    num_cols = len(df_no_nulls.columns)
    print("**********CLEANED NULLS************")
    print(f"Number of rows and columns: {num_rows}, {num_cols}")
    step = 'deleted rows with nulls'
    data_audit, sequence, audit_schema = append_audit_record(data_audit, dataset_name, source, destination, sequence, step, num_rows, num_cols)
    
    df_encoded = one_hot_encode_features(df_no_nulls, train_bool, configuration.SI_MODEL_LOC, configuration.OHE_MODEL_LOC)
    num_rows = df_encoded.count()
    num_cols = len(df_encoded.columns)
    print("********ONE-HOT ENCODING***********")
    print(f"Number of rows and columns: {num_rows}, {num_cols}")
    step = 'one-hot encoding'
    data_audit, sequence, audit_schema = append_audit_record(data_audit, dataset_name, source, destination, sequence, step, num_rows, num_cols)
    
    df_normalized = normalize(df_encoded, train_bool)
    num_rows = df_encoded.count()
    num_cols = len(df_encoded.columns)
    print("************NORMALIZED*************")
    print(f"Number of rows and columns: {num_rows}, {num_cols}")
    step = 'feature normalization'
    data_audit, sequence, audit_schema = append_audit_record(data_audit, dataset_name, source, destination, sequence, step, num_rows, num_cols)
    
    display(df_normalized)
    
    if destination is not None:
        df_encoded.write.mode('overwrite').parquet(destination)

    save_data_audit(data_audit, audit_schema, configuration.DATA_AUDIT_LOC, overwrite=False)
    
    return df_encoded


# COMMAND ----------

# Transform train data set and all test datasets
print('**********************************************')
print('************ 2015 - 2018 Dataset *************')
print('**********************************************')
tranform_and_store('Training Data(2015-2018)', configuration.FINAL_JOINED_DATA_2015_2018, configuration.TRANSFORMED_TRAINING_DATA, True)


# COMMAND ----------

print('**********************************************')
print('*****************2019 Dataset*****************')
print('**********************************************')
tranform_and_store('Test Data(2019)', configuration.FINAL_JOINED_DATA_2019, configuration.TRANSFORMED_2019_DATA, False)


# COMMAND ----------

print('**********************************************')
print('****************2020 Dataset******************')
print('**********************************************')
tranform_and_store('Test Data(2020)', configuration.FINAL_JOINED_DATA_2020, configuration.TRANSFORMED_2020_DATA, False)


# COMMAND ----------

print('**********************************************')
print('*****************2021 Dataset*****************')
print('**********************************************')
tranform_and_store('Test Data(2021)', configuration.FINAL_JOINED_DATA_2021, configuration.TRANSFORMED_2021_DATA, False)


# COMMAND ----------

# Transform toy training data set.
print('**********************************************')
print('***********TOY DATASET (Q1 2015-2018)*********')
print('**********************************************')
tranform_and_store( \
    'Toy Train Data(Q1 2015-2018)', configuration.FINAL_JOINED_DATA_Q1_2015_2018, \
    configuration.TRANSFORMED_Q1_2015_2018_DATA, True)


# COMMAND ----------

# Transform toy test data set.
print('**********************************************')
print('************TOY DATASET (Q1 2019)*************')
print('**********************************************')
tranform_and_store('Toy Test Data(Q1 2019)', configuration.FINAL_JOINED_DATA_Q1_2019, \
                   configuration.TRANSFORMED_Q1_2019_DATA, True)


# COMMAND ----------

def test_storing_audit_info():
    audit_schema = None
    data_audit = [] 
    sequence = 1

    data_audit, sequence, audit_schema = append_audit_record(data_audit, 'test_dataset_name', 'test_source', 'test_destination', sequence, 'test_step', 100, 150)
    save_data_audit(data_audit, audit_schema, f'{configuration.blob_url}/data_audit_test')

test_storing_audit_info()
display(spark.read.parquet(f'{configuration.blob_url}/data_audit_test'))

# COMMAND ----------

def print_summary(location):
    df = spark.read.parquet(location)
    df_filtered = df.select('timestamp', 'dataset_name', 'sequence', 'STEP', 'rows_after', 'cols_after') #\
        #.filter(col('dataset_name') == 'TrainingData(2015-2018)')
    
    display(df_filtered)

print_summary(configuration.DATA_AUDIT_LOC)

# COMMAND ----------

# MAGIC %md # Verify Training Dataset

# COMMAND ----------

training_data = spark.read.parquet(configuration.TRANSFORMED_TRAINING_DATA)
training_data.createOrReplaceTempView('vw_2015_2018')

# COMMAND ----------

training_data_joined = spark.read.parquet(configuration.FINAL_JOINED_DATA_2015_2018)
training_data_joined.createOrReplaceTempView('vw_joined_2015_2018')

# COMMAND ----------

display(training_data.groupby('prior_delayed').count())

# COMMAND ----------

display(training_data.groupby('prior_delayed').count())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT prior_delayed, DEP_DEL15, count(*) as count FROM vw_2015_2018 GROUP BY prior_delayed, DEP_DEL15

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT _CRS_DEPT_HR, count(*) as counts FROM vw_2015_2018 GROUP BY _CRS_DEPT_HR
# MAGIC -- MONTH', 'DAY_OF_WEEK', 'CRS_DEPT_HR', 'DISTANCE_GROUP'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) as counts FROM vw_2015_2018 WHERE MONTH is Null

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT _MONTH, count(*) as counts FROM vw_2015_2018 GROUP BY _MONTH
# MAGIC -- MONTH', 'DAY_OF_WEEK', 'CRS_DEPT_HR', 'DISTANCE_GROUP'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT _DISTANCE_GROUP, count(*) as counts from vw_2015_2018 GROUP by _DISTANCE_GROUP

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTANCE_GROUP, count(*) as counts from vw_joined_2015_2018 GROUP by DISTANCE_GROUP

# COMMAND ----------


