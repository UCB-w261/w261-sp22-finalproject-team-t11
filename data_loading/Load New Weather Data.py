# Databricks notebook source
# MAGIC %md #Initialize Notebook

# COMMAND ----------

import os
import tarfile
from pyspark import SparkFiles
from pyspark.sql.functions import col, split, to_utc_timestamp, count, year, lit, lower
from datetime import datetime


# COMMAND ----------

# Define constants
# Location of raw data.
IATA_TZ_MAP_RAW_LOC = f"{blob_url}/raw/iata_tz_map"
AIRPORTS_CODE_RAW_LOC = f"{blob_url}/raw/airport_codes"
AIRPORTS_MASTER_RAW_LOC = f"{blob_url}/raw/airports"

# Original given sources:
# FLIGHT_RAW_LOC =  "/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*"
# WEATHER_STATIONS_RAW_LOC = "/mnt/mids-w261/datasets_final_project/stations_data/*"
# WEATHER_RAW_LOC = "/mnt/mids-w261/datasets_final_project/weather_data/*"

# New data sources:
FLIGHT_RAW_LOC = f"{blob_url}/raw/flights"
WEATHER_STATIONS_RAW_LOC = f"{blob_url}/raw/stations"
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

# Define constants
# Location of raw data.


# COMMAND ----------

# MAGIC %md # New Weather Data

# COMMAND ----------

# check that there is a temp folder
dbutils.fs.ls('/FileStore/shared_uploads/ram.senth@berkeley.edu')

# COMMAND ----------

# check for tar file in temp folder

# dbutils.fs.ls(f"{blob_url}/downloads/")
# dbutils.fs.cp(f"{blob_url}/downloads/2015.tar.gz", '/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/', True)
dbutils.fs.ls('/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2015')


# COMMAND ----------

def extract_weather_data():
  # Get 2015 data
  # dbutils.fs.mkdirs('/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2015/')
  # dbutils.fs.cp(f"{blob_url}/downloads/2015.tar.gz", '/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2015', True)
  tf_2015 = tarfile.open("/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2015/2015.tar.gz")
  tf_2015.extractall("/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2015/")

  # Get 2016 data
  dbutils.fs.mkdirs('/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2016/')
  dbutils.fs.cp(f"{blob_url}/downloads/2016.tar.gz", '/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2016', True)
  tf_2016 = tarfile.open("/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2016/2016.tar.gz")
  tf_2016.extractall("/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2016/")

  # Get 2017 data
  dbutils.fs.mkdirs('/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2017/')
  dbutils.fs.cp(f"{blob_url}/downloads/2017.tar.gz", '/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2017', True)
  tf_2017 = tarfile.open("/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2017/2017.tar.gz")
  tf_2017.extractall("/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2017/")

  # Get 2018 data
  dbutils.fs.mkdirs('/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2018/')
  dbutils.fs.cp(f"{blob_url}/downloads/2018.tar.gz", '/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2018', True)
  tf_2018 = tarfile.open("/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2018/2018.tar.gz")
  tf_2018.extractall("/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2018/")

  # Get 2019 data
  dbutils.fs.mkdirs('/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2019/')
  dbutils.fs.cp(f"{blob_url}/downloads/2019.tar.gz", '/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2019', True)
  tf_2019 = tarfile.open("/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2019/2019.tar.gz")
  tf_2019.extractall("/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2019/")

  # Get 2020 data
  dbutils.fs.mkdirs('/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2020/')
  dbutils.fs.cp(f"{blob_url}/downloads/2020.tar.gz", '/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2020', True)
  tf_2020 = tarfile.open("/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2020/2020.tar.gz")
  tf_2020.extractall("/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2020/")

  # Get 2021 data
  dbutils.fs.mkdirs('/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2021/')
  dbutils.fs.cp(f"{blob_url}/downloads/2021.tar.gz", '/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2021', True)
  tf_2021 = tarfile.open("/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2021/2021.tar.gz")
  tf_2021.extractall("/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2021/")

extract_weather_data()

# COMMAND ----------

dbutils.fs.ls('/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2020/2020.tar.gz')

# COMMAND ----------

# Cleanup done after extraction of CSSVs.
def delete_unwanted_csvs():
  for fileinfo in dbutils.fs.ls('/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/'):
    if fileinfo.path.endswith('.csv'):
      print(f'Nuking {fileinfo.path}')
      dbutils.fs.rm(fileinfo.path)

def locate_gzips():
  folders = ['2015', '2016', '2017', '2018', '2019', '2020', '2021']
  for folder in folders:
    for fileinfo in dbutils.fs.ls(f'/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/{folder}'):
        if not fileinfo.path.endswith('.csv'):
          print(f'Found {fileinfo.path}')
def move_gzips():
  dbutils.fs.mv('dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2015/2015.tar.gz', 'dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2015.tar.gz')
  dbutils.fs.mv('dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2016/2016.tar.gz', 'dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2016.tar.gz')
  dbutils.fs.mv('dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2017/2017.tar.gz', 'dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2017.tar.gz')
  dbutils.fs.mv('dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2018/2018.tar.gz', 'dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2018.tar.gz')
  dbutils.fs.mv('dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2019/2019.tar.gz', 'dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2019.tar.gz')
  dbutils.fs.mv('dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2020/2020.tar.gz', 'dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2020.tar.gz')
  dbutils.fs.mv('dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2021/2021.tar.gz', 'dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2021.tar.gz')

# move_gzips()
# locate_gzips()
  

# COMMAND ----------

# Copy weather data to raw, excluding data from stations that we are not interested in.
def load_airport_ws_data():
  df = spark.read.parquet(AIRPORTS_WS_LOC)
  df.createOrReplaceTempView("vw_airports_ws")
  return df

def get_required_ws_ids():
  df = load_airport_ws_data()
  return set(list(df.select(lower(col('ws_station_id'))).toPandas()['lower(ws_station_id)']))

def stage_files(years):
    req_ws_ids_set = get_required_ws_ids()
    print(f'{datetime.now().strftime("%H:%M:%S")}: Read {len(req_ws_ids_set)} weather station ids.')
    for year in years:
        counter = 0
        src_location = f'dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/{year}/'
        print(f'{datetime.now().strftime("%H:%M:%S")}: Reading {src_location}')

        for fileinfo in dbutils.fs.ls(src_location):
            ws_id = fileinfo.name.split('.')[0].lower()
            dest_folder = f'dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/weather/reduced/{year}'
            
            if ws_id in req_ws_ids_set:
                # Copy file to new location
                dest_location = f'{dest_folder}/{fileinfo.name}'
                print(f'copying {dest_location}')
                dbutils.fs.cp(fileinfo.path, dest_location)
                counter = counter + 1
        print(f'Copied {counter} files to {dest_folder}')

stage_files(['2015', '2016', '2017', '2018', '2019', '2020', '2021'])


# COMMAND ----------

def read_data(data_location):
#     utc_year = lit(int(year))
  
    df = spark.read.option("recursiveFileLookup", "true")\
        .csv(data_location, header=True, inferSchema=False)\
        .withColumn("_utc_year", year(col('DATE')))
#     display(df.groupBy(col('_utc_year')).count())
    return df

def push_new_weather_data_to_raw_store_all_years():
    data_location = f'dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/weather/reduced/*'
    push_new_weather_data_from(data_location)

def push_new_weather_data_from(location):
    print(f'{datetime.now().strftime("%H:%M:%S")}: Reading out data from {location}')
    df = read_data(location)
    print(f'{datetime.now().strftime("%H:%M:%S")}: Writing out data to {WEATHER_RAW_LOC}')
    df.repartition(col('_utc_year')).write.partitionBy('_utc_year').mode('overwrite').parquet(WEATHER_RAW_LOC)
    print(f'{datetime.now().strftime("%H:%M:%S")}: Done writing out data.')
  
def push_new_weather_data_to_raw_store_per_year(years):
  for cur_year in years:
    data_location = f'dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/weather/reduced/{cur_year}/*'
    push_new_weather_data_from(data_location)
                                         
# push_new_weather_data_to_raw_store(['2015', '2016', '2017', '2018', '2019', '2020', '2021'])
push_new_weather_data_to_raw_store_all_years()
# push_new_weather_data_to_raw_store_all_years()


# COMMAND ----------

# Simple join for weather for SFO airport
def test1():
    data_location = f'dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/weather/reduced/2015/*'
    print(f'{datetime.now().strftime("%H:%M:%S")}: Reading out data')
    df = read_data(data_location)
    df.orderBy(col('DATE'))
    display(df)
  
def test():
    df_airports = spark.read.parquet(AIRPORTS_MASTER_LOC)
    df_airports.createOrReplaceTempView("vw_airports_master")
    df_airports_weather_raw = spark.read.parquet(WEATHER_RAW_LOC)
    df_airports_weather_raw.createOrReplaceTempView("vw_weather_raw")
    
    # Join flights to airport to weather.
    
    sql = """
        SELECT a.iata, year(DATE) as utc_year, count(*) counts
            FROM vw_airports_master as a 
            JOIN vw_weather_raw as b ON a.ws_station_id = b.STATION 
            GROUP BY a.iata, year(DATE)
            ORDER BY a.iata, year(DATE)
    """
    df = spark.sql(sql)
    recs = df.count()
    print(f'# of records: {recs}')
    display(df)

test1()

# COMMAND ----------

def print_count_for(path):
    files = dbutils.fs.ls(path)
    print(f'Found {len(files)} files in {path}')
def print_file_counts():
    path1 = f'dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2015/'
    path2 = f'dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/weather/reduced/2015/'
    
    print_count_for(path1)
    print_count_for(path2)

print_file_counts()
