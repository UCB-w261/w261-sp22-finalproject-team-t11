# Databricks notebook source
# MAGIC %md # Explore Missing Weather Stations

# COMMAND ----------

import os
from pyspark import SparkFiles
from pyspark.sql.functions import col, split, to_utc_timestamp, count, year
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import geopandas as gpd
import seaborn as sns


%matplotlib inline

# COMMAND ----------

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
FLIGHT_PRE_COVID_RAW_LOC =  "/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*"
WEATHER_STATIONS_RAW_LOC = "/mnt/mids-w261/datasets_final_project/stations_data/*"
WEATHER_RAW_LOC = "/mnt/mids-w261/datasets_final_project/weather_data/*"

# New data sources:
FLIGHT_COVID_RAW_LOC = f"{blob_url}/raw/flights_covid"
# WEATHER_STATIONS_RAW_LOC = f"{blob_url}/raw/stations"
WEATHER_RAW_LOC = f"{blob_url}/raw/weather"

# Location of staged data.
AIRPORT_WEATHER_LOC = f"{blob_url}/raw/airport_weather"
AIRPORTS_MASTER_LOC = f"{blob_url}/staged/airports"
AIRPORTS_WS_LOC = f"{blob_url}/staged/airports_weatherstations"
WEATHER_LOC = f"{blob_url}/staged/weather"

# Location of final joined data.
FINAL_JOINED_DATA_ALL = f"{blob_url}/staged/final_joined_all"
# FINAL_JOINED_DATA_TRAINING
# FINAL_JOINED_DATA_VALIDATION
# FINAL_JOINED_DATA_TEST
# FINAL_JOINED_DATA_20_21

SHAPES_BASE_FOLDER = "/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/shapes"

# COMMAND ----------

def load_views_for_weather_station_analysis():
    spark.read.parquet(AIRPORTS_MASTER_LOC).createOrReplaceTempView("vw_airports_master")
    spark.read.parquet(AIRPORTS_WS_LOC).createOrReplaceTempView("vw_airports_ws")
    spark.read.parquet(f'{WEATHER_LOC}/clean_weather_data.parquet').createOrReplaceTempView("vw_weather_cleaned")
    spark.read.parquet(FLIGHT_PRE_COVID_RAW_LOC).createOrReplaceTempView("vw_pre_covid_flight_raw")
    spark.read.parquet(FLIGHT_COVID_RAW_LOC).createOrReplaceTempView("vw_covid_flight_raw")
    spark.read.parquet(FINAL_JOINED_DATA_ALL).createOrReplaceTempView("vw_final_joined")
    spark.read.parquet(WEATHER_RAW_LOC).createOrReplaceTempView("vw_weather_raw")
    spark.read.parquet(WEATHER_STATIONS_RAW_LOC).createOrReplaceTempView("vw_ws_raw")
    spark.sql("SELECT DISTINCT iata FROM (SELECT DISTINCT ORIGIN as iata FROM vw_pre_covid_flight_raw UNION SELECT DISTINCT DEST as iata FROM vw_pre_covid_flight_raw)").createOrReplaceTempView("vw_pre_covid_iatas")
    spark.sql("cache lazy table vw_pre_covid_iatas")
    spark.sql("SELECT DISTINCT iata FROM (SELECT DISTINCT ORIGIN as iata FROM vw_covid_flight_raw UNION SELECT DISTINCT DEST as iata FROM vw_covid_flight_raw)").createOrReplaceTempView("vw_covid_iatas")
    spark.sql("cache lazy table vw_covid_iatas")
        
load_views_for_weather_station_analysis()


# COMMAND ----------

def print_summary_for(era, title):
    if era == 'pre-covid':
        iata_filter = 'iata in(SELECT iata from vw_pre_covid_iatas)'
    elif era == 'covid':
        iata_filter = 'iata in(SELECT iata from vw_covid_iatas)'
    else:
        iata_filter = '1 = 1'
        
    sql = """
        WITH 
            airports_count as (SELECT 'Total Airports' as name, count(*) as count, 1 as seq from vw_airports_master),
            pre_covid_airports as (SELECT 'Pre-Covid Airports' as name, count(*) as count, 2 as seq from vw_pre_covid_iatas),
            covid_airports as (SELECT 'Covid Airports' as name, count(*) as count, 3 as seq from vw_covid_iatas),
            airport_ws_count as (
                SELECT 'Weather Stations Near Airports*' as name, count(*) as count, 4 as seq 
                FROM vw_airports_ws WHERE """ + iata_filter + """
            ),
            ws_with_weather_raw as (
                SELECT 
                    'Weather Stations Near Airports With Raw Weather Data' as name, 
                    count(DISTINCT STATION) as count, 5 as seq 
                    FROM vw_weather_raw
            ),
            ws_with_weather_cleaned as (
                SELECT 'Weather Stations Near Airports With Cleaned Weather Data' as name, 
                    count(DISTINCT STATION) as count, 
                    6 as seq 
                FROM vw_weather_cleaned
            ),
            airports_with_raw_weather_missing as (
                SELECT 'Airports Missing Some Raw Weather Data*' as name, 
                    count(DISTINCT iata) as count, 7 as seq
                FROM vw_airports_ws 
                WHERE 
                    """ + iata_filter + """
                    AND vw_airports_ws.ws_station_id not in (SELECT DISTINCT STATION from vw_weather_raw)
            ),
            airports_with_atleast_one_ws as (
                SELECT 'Airports With Raw Weather Data From Atleast One Station*' as name, 
                    count(DISTINCT iata) as count, 8 as seq 
                FROM vw_airports_ws 
                WHERE 
                    """ + iata_filter + """
                    AND vw_airports_ws.ws_station_id in (SELECT DISTINCT STATION from vw_weather_raw)
            ),
            airports_with_cleaned_weather_missing as (
                SELECT 'Airports Missing Some Cleaned Weather Data*' as name, 
                    count(DISTINCT iata) as count, 9 as seq
                FROM vw_airports_ws 
                WHERE 
                    """ + iata_filter + """ 
                    AND vw_airports_ws.ws_station_id not in 
                        (SELECT DISTINCT STATION from vw_weather_cleaned WHERE STATION is not NULL)
            ),
            airports_with_atleast_one_clean_weather as (
                SELECT 'Airports With Cleaned Weather Data From Atleast One Station*' as name, 
                    count(DISTINCT iata) as count, 10 as seq 
                FROM vw_airports_ws 
                WHERE 
                    """ + iata_filter + """ 
                    AND vw_airports_ws.ws_station_id in 
                        (SELECT DISTINCT STATION from vw_weather_cleaned WHERE STATION is not NULL))
                    
        SELECT seq, name, count FROM airports_count 
            UNION SELECT seq, name, count FROM pre_covid_airports 
            UNION SELECT seq, name, count FROM covid_airports
            UNION SELECT seq, name, count FROM airport_ws_count
            UNION SELECT seq, name, count FROM ws_with_weather_raw 
            UNION SELECT seq, name, count FROM ws_with_weather_cleaned
            UNION SELECT seq, name, count FROM airports_with_raw_weather_missing
            UNION SELECT seq, name, count FROM airports_with_atleast_one_ws
            UNION SELECT seq, name, count FROM airports_with_cleaned_weather_missing
            UNION SELECT seq, name, count FROM airports_with_atleast_one_clean_weather
            ORDER BY seq
    """
    print(title)
    display(spark.sql(sql))
        
print_summary_for('all', 'Analysis of weather for all airports')
print_summary_for('pre-covid', 'Analysis of weather for airports with pre-covid flights')
print_summary_for('covid', 'Analysis of weather for airports with covid flights')


# COMMAND ----------

# MAGIC %sql
# MAGIC WITH stations_in_raw as (SELECT DISTINCT STATION from vw_weather_raw),
# MAGIC stations_in_cleaned as (SELECT DISTINCT STATION from vw_weather_cleaned WHERE STATION is not NULL)
# MAGIC SELECT a.STATION as `Station(raw)`, b.STATION as `Station(cleaned)` 
# MAGIC   FROM stations_in_raw a FULL OUTER JOIN stations_in_cleaned b ON a.STATION = b.STATION
# MAGIC WHERE a.STATION is NULL or b.STATION is NULL

# COMMAND ----------

def missing_stations():
    stations = '70316025624, 72064354940, 72255012912, 72445703938, 72451013985, 72462023061, 72480023157, 72512704726, 72514014778, 72531603887, 72547804920, 72548514940, 72549094933, 72666024029, 72672024061, 72781024243, 72782594239, 72784624160, 72785794129'
    display(spark.sql(f"SELECT * from vw_weather_cleaned where STATION in ({stations})"))
            
missing_stations()
            


# COMMAND ----------


