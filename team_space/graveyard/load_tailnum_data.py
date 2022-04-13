# Databricks notebook source
import os
from pyspark import SparkFiles
from pyspark.sql.functions import col, when, to_utc_timestamp, to_timestamp, year, date_trunc, split, regexp_replace, array_max, length, substring, greatest, minute, hour, expr, count, countDistinct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import geopandas as gpd
import seaborn as sns

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
# AIRPORT_WEATHER_LOC = f"{blob_url}/raw/airport_weather"
AIRPORTS_MASTER_LOC = f"{blob_url}/staged/airports"
AIRPORTS_WS_LOC = f"{blob_url}/staged/airports_weatherstations"
WEATHER_LOC = f"{blob_url}/staged/weather"
CLEAN_WEATHER_LOC = f'{WEATHER_LOC}/clean_weather_data.parquet'

# Location of final joined data.
FINAL_JOINED_DATA_ALL = f"{blob_url}/staged/final_joined_all"
# FINAL_JOINED_DATA_TRAINING
# FINAL_JOINED_DATA_VALIDATION
# FINAL_JOINED_DATA_TEST
# FINAL_JOINED_DATA_20_21

SHAPES_BASE_FOLDER = "/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/shapes"

tail_num_filepath = "dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/new_data/aircrafts/mapped_tail_num.csv"

# COMMAND ----------

tail_num_data = spark.read.options(inferSchema = True, header = True).csv(tail_num_filepath)
display(tail_num_data)

# COMMAND ----------

tail_num_data.printSchema()

# COMMAND ----------

keep_cols = ['tail_num', 'year', 'year_mfr', 'certification', 'type_aircraft', 'type_engine', 'mode_s_code_hex', 'no-eng', 'no-seats', 'ac-weight', 'type_aircraft', 'type_engine', 'mfr', 'model', 'no-seats', 'ac-weight']

# model will need more cleaning before one-hot encoding
# idea: remove details after the last '-' in the model name

drop_cols = ['_c0', 'mfr_mdl_code', 'eng_mfr_mdl', 'tail_number', 'serial_number', 'type_registrant', 'name', 'street', 'street2', 'city', 'state', 'zip_code', 'region', 'county', 'country', 'last_action_date', 'cert_issue_date', 'certification', 'status_code', 'air_worth_date', 'mode_s_code', 'mode_s_code_hex', 'fract_owner', 'other_names(1)', 'other_names(2)', 'other_names(3)', 'other_names(4)', 'other_names(5)', 'expiration_date', 'unique_id', 'kit_mfr', '_kit_model', 'ac-cat', 'type-acft', 'type-eng', 'build-cert-ind', 'speed']
# drop number of engines since this info is likely incorporated in the model anyway
# drop speed because there are a lot of zeros

# COMMAND ----------

display(tail_num_data.select('no-eng').groupBy('no-eng').count())

# COMMAND ----------


