# Databricks notebook source
import os
import tarfile
from pyspark import SparkFiles
from pyspark.sql.functions import col, split, to_utc_timestamp, count, year, lit

blob_container = "w261team11" # The name of your container created in https://portal.azure.com
storage_account = "w261sa" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261team11" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261team11key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)


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
FLIGHT_RAW_LOC =  "/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*"
# WEATHER_STATIONS_RAW_LOC = "/mnt/mids-w261/datasets_final_project/stations_data/*"
# WEATHER_RAW_LOC = "/mnt/mids-w261/datasets_final_project/weather_data/*"

# New data sources:
FLIGHT_COVID_RAW_LOC = f"{blob_url}/raw/flights_covid"
WEATHER_STATIONS_RAW_LOC = f"{blob_url}/raw/stations"
WEATHER_RAW_LOC = f"{blob_url}/raw/weather"

# Location of staged data.
AIRPORT_WEATHER_LOC = f"{blob_url}/raw/airport_weather"
AIRPORTS_MASTER_LOC = f"{blob_url}/staged/airports"
AIRPORTS_WS_LOC = f"{blob_url}/staged/airports_weatherstations"
WEATHER_LOC = f"{blob_url}/staged/weather"

# Location of final joined data.
FINAL_JOINED_DATA_ALL = f"{blob_url}/staged/weather/final_joined_all"
# FINAL_JOINED_DATA_TRAINING
# FINAL_JOINED_DATA_VALIDATION
# FINAL_JOINED_DATA_TEST
# FINAL_JOINED_DATA_20_21

SHAPES_BASE_FOLDER = "/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/shapes"

# COMMAND ----------

# MAGIC %md # New Flights Data

# COMMAND ----------

# check for tar file in temp folder

# dbutils.fs.ls(f"{blob_url}/downloads/")
# dbutils.fs.cp(f"{blob_url}/downloads/2015.tar.gz", '/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/', True)
dbutils.fs.ls('/FileStore/shared_uploads/ram.senth@berkeley.edu/new_data/flights')


# COMMAND ----------

def explore():
    df = spark.read.csv('dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/new_data/flights/On_Time_Reporting_Carrier_On_Time_Performance__1987_present__2020_1.csv', header=True, inferSchema=True)
    print(df.schema)
    
def ingest():
    src_files = dbutils.fs.ls('/FileStore/shared_uploads/ram.senth@berkeley.edu/new_data/flights')
    for fileinfo in src_files:
        print(f'Reading flights data from {fileinfo}')
        df = spark.read.csv(fileinfo.path, header=True, inferSchema=True).withColumn("_local_year", col('Year'))
        print(f'Appending to existing data at {FLIGHT_RAW_LOC}')
        df.write.mode('append').partitionBy('_local_year').parquet(FLIGHT_RAW_LOC)

        
ingest()

# COMMAND ----------

# MAGIC %md ## Do we have additional airports?
# MAGIC We have 17 new airports showing up in the 20-21 data. Need to ensure that we have weather stations and timezones identified for these airports as well.

# COMMAND ----------

def create_views():
    df_pre_covid_flights = spark.read.parquet(FLIGHT_RAW_LOC)
    df_pre_covid_flights.createOrReplaceTempView("pre_covid_flights_vw")
    df_covid_flights = spark.read.parquet(FLIGHT_COVID_RAW_LOC)
    df_covid_flights.createOrReplaceTempView("covid_flights_vw")

create_views()

# COMMAND ----------

# MAGIC %sql
# MAGIC with iatas_pre_covid as (
# MAGIC     SELECT DISTINCT ORIGIN as iata FROM pre_covid_flights_vw UNION SELECT DISTINCT DEST as iata FROM pre_covid_flights_vw
# MAGIC ),
# MAGIC iatas_covid as (
# MAGIC     SELECT DISTINCT ORIGIN as iata FROM covid_flights_vw UNION SELECT DISTINCT DEST as iata FROM covid_flights_vw
# MAGIC )
# MAGIC SELECT a.iata as a_iata, b.iata as b_iata from iatas_pre_covid a FULL OUTER JOIN iatas_covid b ON a.iata = b.iata WHERE a.iata is null or b.iata is null

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT YEAR, count(*) from covid_flights_vw group by YEAR

# COMMAND ----------

# MAGIC %sql
# MAGIC with iatas_pre_covid as (
# MAGIC     SELECT DISTINCT ORIGIN as iata FROM pre_covid_flights_vw UNION SELECT DISTINCT DEST as iata FROM pre_covid_flights_vw
# MAGIC ),
# MAGIC iatas_covid as (
# MAGIC     SELECT DISTINCT ORIGIN as iata FROM covid_flights_vw UNION SELECT DISTINCT DEST as iata FROM covid_flights_vw
# MAGIC ),
# MAGIC all_iatas as (SELECT DISTINCT iata FROM iatas_pre_covid UNION SELECT DISTINCT iata FROM iatas_covid)
# MAGIC SELECT distinct iata from all_iatas ORDER by iata

# COMMAND ----------


