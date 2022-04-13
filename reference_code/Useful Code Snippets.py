# Databricks notebook source
# MAGIC %md # Configuration

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

SHAPES_BASE_FOLDER = "/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/shapes"

# COMMAND ----------

# Data location

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


# COMMAND ----------

# MAGIC %md # Loading data

# COMMAND ----------

# Loading source/raw data.

# Load the weather stations data
spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*").createOrReplaceTempView("stations_vw_src")

# Airports master with one row per airport for all 383 airports found in Covid + precovid flights data.
spark.read.parquet(AIRPORTS_MASTER_LOC).createOrReplaceTempView("vw_airports_master")

# Airports - weather station mapping - 
spark.read.parquet(AIRPORTS_WS_LOC).createOrReplaceTempView("vw_airports_ws")
spark.read.parquet(f'{WEATHER_LOC}/clean_weather_data.parquet').createOrReplaceTempView("vw_weather_cleaned")
spark.read.parquet(FLIGHT_RAW_LOC).createOrReplaceTempView("vw_flight_raw")
spark.read.parquet(FINAL_JOINED_DATA_ALL).createOrReplaceTempView("vw_final_joined")
spark.read.parquet(WEATHER_RAW_LOC).createOrReplaceTempView("vw_weather_raw")
spark.read.parquet(WEATHER_STATIONS_RAW_LOC).createOrReplaceTempView("vw_ws_raw")



# COMMAND ----------

# Loading 

