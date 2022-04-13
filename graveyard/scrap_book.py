# Databricks notebook source
# MAGIC %md # Data Exploration

# COMMAND ----------

# MAGIC %md ## Setup and Load Data

# COMMAND ----------

# MAGIC %md ###Setup

# COMMAND ----------

import os
from pyspark import SparkFiles
from pyspark.sql.functions import col, split, to_utc_timestamp, count
import matplotlib.pyplot as plt
import numpy as np

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
# 2015 - 2018 flights data used for training. This needs to be further split into training-vaildation 
# sets for cross validation
FINAL_JOINED_DATA_TRAINING = f"{blob_url}/staged/final_joined_training"

# 2019 flights data used for testing model.
FINAL_JOINED_DATA_TEST = f"{blob_url}/staged/final_joined_testing"

# 2020-2021 covid era flights for additional exploration/testing.
FINAL_JOINED_DATA_20_21 = f"{blob_url}/staged/final_joined_2020_2021"

SHAPES_BASE_FOLDER = "/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/shapes"

# COMMAND ----------

# MAGIC %md ###Get Delay Summary Data

# COMMAND ----------

def copy_delay_causes_to_blob_store():
    df_delay = spark.read.csv("/user/ram.senth@berkeley.edu/delay_cause.csv", header=True, inferSchema= True)
    df_delay.write.mode('overwrite').parquet(f"{blob_url}/raw/delay_summary")
# Need to run this just once.
# copy_delay_causes_to_blob_store()

# COMMAND ----------

# MAGIC %md ### Get Airport Timezones
# MAGIC Source: http://www.fresse.org/dateutils/tzmaps.html

# COMMAND ----------

def copy_airport_tz_to_blob_store():
    df_delay = spark.read.csv("/user/ram.senth@berkeley.edu/iata_tzmap.csv", header=True, inferSchema= True)
    df_delay.write.mode('overwrite').parquet(f"{blob_url}/raw/iata_tz_map")
# Need to run this just once.
# copy_airport_tz_to_blob_store()

# COMMAND ----------

# MAGIC %md ### Get Airport Codes
# MAGIC 
# MAGIC Source: https://datahub.io/core/airport-codes#resource-airport-codes

# COMMAND ----------

def copy_airport_codes_to_blob_store():
    df_delay = spark.read.csv("/user/ram.senth@berkeley.edu/airport-codes.csv", header=True, inferSchema= True)
    df_delay.write.mode('overwrite').parquet(f"{blob_url}/raw/airport_codes")
# Need to run this just once.
# copy_airport_codes_to_blob_store()

# COMMAND ----------

# MAGIC %md ### Load All Data

# COMMAND ----------

# Load Data

# Load delay summary data and register a temp view.
df_delay_summary = spark.read.parquet(f"{blob_url}/raw/delay_summary")
df_delay_summary.createOrReplaceTempView("delay_summary_vw")

# Load airports data and temp view.
df_airport_codes = spark.read.parquet(f"{blob_url}/raw/airport_codes")
df_airport_codes.createOrReplaceTempView("airport_codes_vw_src")

# Load airport tz data and temp view.
df_airport_tz_map = spark.read.parquet(f"{blob_url}/raw/iata_tz_map")
df_airport_tz_map.createOrReplaceTempView("df_airport_tz_map_vw_src")

# Load 3m flights data
df_airlines_3m = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
df_airlines_3m.createOrReplaceTempView("airlines_3m_vw_src")

# Load all flights data. Big dataset. Be warned.
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*")
df_airlines.createOrReplaceTempView("airlines_vw_src")

# Load weather data for Q1 2015
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")
df_weather.createOrReplaceTempView("weather_vw_src")

# Load the weather stations data
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
df_stations.createOrReplaceTempView("stations_vw_src")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT FL_DATE, CRS_DEP_TIME, to_date(FL_DATE), to_timestamp(concat(FL_DATE, ' ', CRS_DEP_TIME, '00'), 'yyyy-MM-dd Hmmss') as local_ts, ORIGIN, iana_tz_name 
# MAGIC from airlines_vw_src a join df_airport_tz_map_vw_src b on a.ORIGIN = b.iata_code

# COMMAND ----------

def check_for_dups():
    df = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*").filter('cancelled <> 1 and TAIL_NUM is not null')
    print(f'Original count: {df.count()}')
    print(f'Distinct count: {df.distinct().count()}')

def check_for_null_values():
    df = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*").filter('cancelled <> 1')
    df.cache()
#     print(f"TAIL_NUM null for {df.filter('TAIL_NUM is null').count()}") # 0
#     print(f"DEP_DEL15 null for {df.filter('DEP_DEL15 is null').count()}") # 9000+
    display(df.filter('DEP_DEL15 is null'))

check_for_null_values()
# check_for_dups() # Original count: 62513788, Distinct count: 31256894

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT distinct YEAR, OP_CARRIER_FL_NUM, TAIL_NUM, ORIGIN, DEST, CRS_DEP_TIME, DEP_TIME, DEP_DELAY, DEP_DEL15, CRS_ARR_TIME, ARR_TIME, ARR_DELAY, ARR_DEL15, CANCELLED, DIVERTED 
# MAGIC FROM airlines_vw_src
# MAGIC WHERE TAIL_NUM = 'N937SW' and FL_DATE = '2015-06-16'

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH main_tbl as (
# MAGIC SELECT a.*,
# MAGIC   to_utc_timestamp(concat(
# MAGIC     FL_DATE, ' ', LEFT(RIGHT(concat('0', CRS_DEP_TIME), 4), 2), ':', RIGHT(RIGHT(concat('0', CRS_DEP_TIME), 4), 2), ':00'
# MAGIC   ), 'PST'/*iana_tz_name*/) as utc_ts /*, iana_tz_name */
# MAGIC FROM airlines_vw_src a 
# MAGIC   /*JOIN df_airport_tz_map_vw_src b ON a.ORIGIN = b.iata_code*/),
# MAGIC dup_flights as (
# MAGIC   SELECT ORIGIN, TAIL_NUM, FL_DATE, CRS_DEP_TIME, count(*) as counts 
# MAGIC   FROM main_tbl group by ORIGIN, TAIL_NUM, FL_DATE, CRS_DEP_TIME
# MAGIC ) 
# MAGIC SELECT * from dup_flights where counts > 1 and ORIGIN = 'SFO'
# MAGIC /*SELECT * FROM main_tbl WHERE FL_DATE in ('2019-05-25') AND origin = 'ADK' ORDER BY CRS_DEP_TIME*/

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH main_tbl as (
# MAGIC SELECT FL_DATE, CRS_DEP_TIME, 
# MAGIC   to_utc_timestamp(concat(
# MAGIC     FL_DATE, ' ', LEFT(RIGHT(concat('0', CRS_DEP_TIME), 4), 2), ':', RIGHT(RIGHT(concat('0', CRS_DEP_TIME), 4), 2), ':00'
# MAGIC   ), iana_tz_name) as utc_ts, ORIGIN as origin, iana_tz_name 
# MAGIC FROM airlines_vw_src a 
# MAGIC   JOIN df_airport_tz_map_vw_src b ON a.ORIGIN = b.iata_code)
# MAGIC SELECT * FROM (
# MAGIC   SELECT *, row_number() OVER (PARTITION BY origin, FL_DATE ORDER BY origin, FL_DATE, CRS_DEP_TIME) as rn FROM main_tbl WHERE FL_DATE in ('2019-05-25', '2019-12-25')
# MAGIC ) tmp WHERE rn = 1

# COMMAND ----------

display(df_airlines)


# COMMAND ----------

# Load the final airports master.
def load_airport_master():
    sql = """
        WITH 
          iatas as (
            SELECT DISTINCT ORIGIN as iata FROM airlines_vw UNION SELECT DISTINCT DEST as iata FROM airlines_vw_src
          ),
          iatas_of_interest as (SELECT DISTINCT iata FROM iatas),
          airport_codes as (
            SELECT type, name, iso_country, gps_code as icao_code, iata_code, 
                local_code, municipality, coordinates, elevation_ft, 
                round(cast(split(coordinates, ',')[0] as double), 1) as airport_lat, 
                round(cast(split(coordinates, ',')[1] as double), 1) as airport_lon
              FROM airport_codes_vw_src
          ),
          stations as (
            SELECT station_id, neighbor_id, neighbor_call, neighbor_name, distance_to_neighbor, 
                cast(lat as double) as lat, cast(lon as double) as lon
              FROM stations_vw_src
          ),
          airports as (
            SELECT ioi.iata, ac.icao_code as iaco, ac.type, ac.name, ac.municipality, 
                ac.iso_country, s.station_id as ws_id, s.neighbor_name as ws_neighbor_name, 
                s.distance_to_neighbor as ws_distance, ac.elevation_ft, s.lat as station_lat, s.lon as station_lon, 
                coordinates, airport_lat, airport_lon 
              FROM iatas_of_interest ioi 
              LEFT OUTER JOIN airport_codes ac ON ioi.iata = ac.iata_code
              LEFT OUTER JOIN stations s ON ac.icao_code = s.neighbor_call
          )
        SELECT * FROM airports
    """
    df_airports = spark.sql(sql)
    df_airports.cache()
    df_airports.createOrReplaceTempView("airports_vw")

load_airport_master()

# COMMAND ----------

# MAGIC %sql
# MAGIC with v_one as(
# MAGIC SELECT type, name, iso_country, gps_code as icao_code, iata_code, 
# MAGIC                 local_code, municipality, coordinates, elevation_ft, 
# MAGIC                 round(cast(split(coordinates, ',')[0] as double), 0) as airport_lon, 
# MAGIC                 round(cast(split(coordinates, ',')[1] as double), 0) as airport_lat
# MAGIC               FROM airport_codes_vw where iata_code = 'SFO'), 
# MAGIC v_two as (SELECT station_id, neighbor_id, neighbor_call, neighbor_name, distance_to_neighbor, 
# MAGIC round(cast(lat as double), 0) as lat, round(cast(lon as double), 0) as lon
# MAGIC FROM stations_vw WHERE neighbor_call in ('KSFO') and distance_to_neighbor < 25)
# MAGIC SELECT v_one.*, v_two.* from v_one join v_two WHERE airport_lon  = lon and airport_lat = lat

# COMMAND ----------

# MAGIC %md ## Summary of Flight Delays (2015-2019)
# MAGIC Data Source: https://www.transtats.bts.gov/OT_Delay/ot_delaycause1.asp?qv52ynB=qn6n&20=E

# COMMAND ----------

display(df_delay_summary)

# COMMAND ----------

def aggregate_delays():
    sql = """
        SELECT 
            sum(arr_flights) as total_flights,
            sum(arr_del15) as delayed_flights,
            sum(arr_flights) - sum(arr_del15) - sum(arr_cancelled) -sum(arr_diverted) as on_time_flights,
            round(sum(carrier_ct), 0) as carrier_delay_counts, 
            round(sum(weather_ct), 0) as weather_delay_counts, 
            round(sum(nas_ct), 0) as nas_delay_counts, 
            round(sum(security_ct), 0) as security_delay_counts, 
            round(sum(late_aircraft_ct), 0) as late_aircraft_delay_counts, 
            sum(carrier_delay) as carrier_delay_mins, 
            sum(weather_delay) as weather_delay_mins, 
            sum(nas_delay) as nas_delay_mins, 
            sum(security_delay) as security_delay_mins, 
            sum(late_aircraft_delay) as late_aircraft_delay_mins,
            sum(arr_cancelled) as arrival_cancelled,
            sum(arr_diverted) as arrival_diverted
        FROM delay_summary_vw
    """
#     sql = "DESCRIBE delay_summary_vw"
#     sql = "SELECT sum(carrier_delay) from delay_summary_vw"
    result_df = spark.sql(sql)
    return result_df

def aggregate_by_moy_delays():
    sql = """
        SELECT month, 
            sum(arr_flights) as total_flights,
            sum(arr_del15) as delayed_flights,
            sum(arr_flights) - sum(arr_del15) - sum(arr_cancelled) -sum(arr_diverted) as on_time_flights,
            round(sum(carrier_ct), 0) as carrier_delay_counts, 
            round(sum(weather_ct), 0) as weather_delay_counts, 
            round(sum(nas_ct), 0) as nas_delay_counts, 
            round(sum(security_ct), 0) as security_delay_counts, 
            round(sum(late_aircraft_ct), 0) as late_aircraft_delay_counts, 
            sum(carrier_delay) as carrier_delay_mins, 
            sum(weather_delay) as weather_delay_mins, 
            sum(nas_delay) as nas_delay_mins, 
            sum(security_delay) as security_delay_mins, 
            sum(late_aircraft_delay) as late_aircraft_delay_mins,
            sum(arr_cancelled) as arrival_cancelled,
            sum(arr_diverted) as arrival_diverted
        FROM delay_summary_vw
        GROUP BY month
    """
#     sql = "DESCRIBE delay_summary_vw"
#     sql = "SELECT sum(carrier_delay) from delay_summary_vw"
    result_df = spark.sql(sql)
    return result_df

# COMMAND ----------

# ORD specific data
def ord_aggregate_by_moy_delays():
    sql = """
        SELECT month, 
            sum(arr_flights) as total_flights,
            sum(arr_del15) as delayed_flights,
            sum(arr_flights) - sum(arr_del15) - sum(arr_cancelled) -sum(arr_diverted) as on_time_flights,
            round(sum(carrier_ct), 0) as carrier_delay_counts, 
            round(sum(weather_ct), 0) as weather_delay_counts, 
            round(sum(nas_ct), 0) as nas_delay_counts, 
            round(sum(security_ct), 0) as security_delay_counts, 
            round(sum(late_aircraft_ct), 0) as late_aircraft_delay_counts, 
            sum(carrier_delay) as carrier_delay_mins, 
            sum(weather_delay) as weather_delay_mins, 
            sum(nas_delay) as nas_delay_mins, 
            sum(security_delay) as security_delay_mins, 
            sum(late_aircraft_delay) as late_aircraft_delay_mins,
            sum(arr_cancelled) as arrival_cancelled,
            sum(arr_diverted) as arrival_diverted
        FROM delay_summary_vw
        WHERE airport = 'ORD'
        GROUP BY month
    """
#     sql = "DESCRIBE delay_summary_vw"
#     sql = "SELECT sum(carrier_delay) from delay_summary_vw"
    result_df = spark.sql(sql)
    return result_df

display(ord_aggregate_by_moy_delays())

# COMMAND ----------

display(aggregate_by_moy_delays())

# COMMAND ----------

display(spark.sql("SELECT distinct airport FROM delay_summary_vw"))

# COMMAND ----------

# By airport
def by_airport_delays():
    sql = """
        SELECT 
            airport, airport_name,
            (sum(arr_flights) - sum(arr_del15) - sum(arr_cancelled) -sum(arr_diverted)) / sum(arr_flights) as percentage_on_time,
            sum(arr_flights) as total_flights,
            sum(arr_del15) as delayed_flights,
            sum(arr_flights) - sum(arr_del15) - sum(arr_cancelled) -sum(arr_diverted) as on_time_flights,
            round(sum(carrier_ct), 0) as carrier_delay_counts, 
            round(sum(weather_ct), 0) as weather_delay_counts, 
            round(sum(nas_ct), 0) as nas_delay_counts, 
            round(sum(security_ct), 0) as security_delay_counts, 
            round(sum(late_aircraft_ct), 0) as late_aircraft_delay_counts, 
            sum(carrier_delay) as carrier_delay_mins, 
            sum(weather_delay) as weather_delay_mins, 
            sum(nas_delay) as nas_delay_mins, 
            sum(security_delay) as security_delay_mins, 
            sum(late_aircraft_delay) as late_aircraft_delay_mins,
            sum(arr_cancelled) as arrival_cancelled,
            sum(arr_diverted) as arrival_diverted
        FROM delay_summary_vw
        GROUP BY airport, airport_name
        HAVING total_flights > 100000
    """
#     sql = "DESCRIBE delay_summary_vw"
#     sql = "SELECT sum(carrier_delay) from delay_summary_vw"
    result_df = spark.sql(sql)
    return result_df
display(by_airport_delays())

# COMMAND ----------

# By Airlines
def by_carrier_delays():
    sql = """
        SELECT 
            carrier, carrier_name,
            (sum(arr_flights) - sum(arr_del15) - sum(arr_cancelled) -sum(arr_diverted)) / sum(arr_flights) as percentage_on_time,
            sum(arr_flights) as total_flights,
            sum(arr_del15) as delayed_flights,
            sum(arr_flights) - sum(arr_del15) - sum(arr_cancelled) -sum(arr_diverted) as on_time_flights,
            round(sum(carrier_ct), 0) as carrier_delay_counts, 
            round(sum(weather_ct), 0) as weather_delay_counts, 
            round(sum(nas_ct), 0) as nas_delay_counts, 
            round(sum(security_ct), 0) as security_delay_counts, 
            round(sum(late_aircraft_ct), 0) as late_aircraft_delay_counts, 
            sum(carrier_delay) as carrier_delay_mins, 
            sum(weather_delay) as weather_delay_mins, 
            sum(nas_delay) as nas_delay_mins, 
            sum(security_delay) as security_delay_mins, 
            sum(late_aircraft_delay) as late_aircraft_delay_mins,
            sum(arr_cancelled) as arrival_cancelled,
            sum(arr_diverted) as arrival_diverted
        FROM delay_summary_vw
        GROUP BY carrier, carrier_name
    """
    result_df = spark.sql(sql)
    return result_df
display(by_carrier_delays())

# COMMAND ----------

# MAGIC %md #Identifying Airport Weather Stations
# MAGIC [ICAO](https://en.wikipedia.org/wiki/ICAO_airport_code) (Internationl Civil Aviation Organization) Code: The neighbor_call field in stations refers to ICAO code. This can be used for join?
# MAGIC 
# MAGIC [IATA](https://en.wikipedia.org/wiki/IATA_airport_code) Code: The 3 letter code used in airlines data
# MAGIC 
# MAGIC airport_codes has ICAO to IATA mapping.

# COMMAND ----------

display(df_airport_codes)

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH 
# MAGIC   iatas as (
# MAGIC     SELECT DISTINCT ORIGIN as iata FROM airlines_vw UNION SELECT DISTINCT DEST as iata FROM airlines_vw
# MAGIC   ),
# MAGIC   iatas_of_interest as (SELECT DISTINCT iata FROM iatas)
# MAGIC SELECT count(*) FROM iatas_of_interest

# COMMAND ----------

def load_airport_weatherstations():
    sql = """
        WITH 
          iatas as (
            SELECT DISTINCT ORIGIN as iata FROM airlines_vw UNION SELECT DISTINCT DEST as iata FROM airlines_vw
          ),
          iatas_of_interest as (SELECT DISTINCT iata FROM iatas),
          airport_codes as (
            SELECT type, name, iso_country, gps_code as icao_code, iata_code, local_code, municipality, elevation_ft, coordinates
              FROM airport_codes_vw
          ),
          stations as (
            SELECT station_id, neighbor_id, neighbor_call, neighbor_name, distance_to_neighbor 
              FROM stations_vw
              WHERE distance_to_neighbor <= 1
          ),
          airports as (
            SELECT ioi.iata, ac.icao_code as iaco, ac.type, ac.name, ac.municipality, ac.iso_country, 
                s.station_id as ws_id, s.neighbor_name as ws_neighbor_name, 
                s.distance_to_neighbor as ws_distance, ac.elevation_ft, 
                cast(split(coordinates, ',')[0] as double) as airport_lon, cast(split(coordinates, ',')[1] as double) as airport_lat 
              FROM iatas_of_interest ioi 
              LEFT OUTER JOIN airport_codes ac ON ioi.iata = ac.iata_code
              LEFT OUTER JOIN stations s ON ac.icao_code = s.neighbor_call
          )
        SELECT * FROM airports
    """

    df_airports_with_ws = spark.sql(sql)
    df_airports_with_ws.cache()
    df_airports_with_ws.createOrReplaceTempView("airports_ws_vw")

load_airport_weatherstations()

# COMMAND ----------

def ws_of_interest():
    sql = """
        SELECT ws_id 
        FROM airports_ws_vw 
        WHERE ws_id is not null"""
    display(spark.sql(sql))
    
ws_of_interest()

# COMMAND ----------

def dummy():
    sql = """
        SELECT iata, iaco, cast(split(coordinates, ',')[0] as double) as long, cast(split(coordinates, ',')[1] as double) as lat 
        FROM airports_ws_vw 
        WHERE ws_id is null OR iata='SFO' """
    display(spark.sql(sql))
dummy()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from stations_vw 
# MAGIC WHERE neighbor_call in ('KSFO', 'KSJC') AND distance_to_neighbor < 10
# MAGIC ORDER BY neighbor_call, distance_to_neighbor

# COMMAND ----------

airports_missing_weather = [
    ('PSE', 'TJPS'), 
    ('PPG', 'NSTU'), 
    ('OGS', 'KOGS'), 
    ('SPN', 'PGSN'), 
    ('SJU', 'TJSJ'), 
    ('TKI', '57A'), 
    ('GUM', 'PGUM'), 
    ('XWA', 'KXWA')]


# COMMAND ----------

# MAGIC %sql
# MAGIC WITH summary as (
# MAGIC   SELECT CONCAT(ORIGIN, '-', DEST) as route, 'F' as direction, count(*) as counts
# MAGIC     FROM airlines_vw 
# MAGIC     WHERE ORIGIN in ('PSE', 'PPG', 'OGS', 'SPN', 'SJU', 'TKI', 'GUM', 'XWA')
# MAGIC     GROUP BY ORIGIN, DEST
# MAGIC   UNION SELECT CONCAT(DEST, '-', ORIGIN) as route, 'R' as direction, count(*) as counts
# MAGIC     FROM airlines_vw 
# MAGIC     WHERE DEST in ('PSE', 'PPG', 'OGS', 'SPN', 'SJU', 'TKI', 'GUM', 'XWA')
# MAGIC     GROUP BY DEST, ORIGIN
# MAGIC ) SELECT * from summary ORDER by route, direction

# COMMAND ----------

# MAGIC %md ## Track Airplanes

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT TAIL_NUM, count(*) as counts FROM airlines_3m_vw
# MAGIC   WHERE ORIGIN = 'ORD'
# MAGIC   GROUP BY TAIL_NUM
# MAGIC   ORDER BY counts desc

# COMMAND ----------

def tail_nums_to_explore(n=10):
    sql = """
        SELECT TAIL_NUM, count(*) as counts FROM airlines_3m_vw
          WHERE ORIGIN = 'ORD'
          GROUP BY TAIL_NUM
    """
    tail_nums_df = spark.sql(sql)
    return tail_nums_df.rdd.map(lambda row: (row["TAIL_NUM"], row["counts"])).takeSample(False, n)
    
print(tail_nums_to_explore(10))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT YEAR, QUARTER, MONTH, DAY_OF_MONTH, DAY_OF_WEEK, FL_DATE, TAIL_NUM, 
# MAGIC   OP_CARRIER_FL_NUM, ORIGIN, DEST, CRS_DEP_TIME, DEP_TIME, CRS_ARR_TIME, ARR_TIME, 
# MAGIC   DEP_DELAY, DEP_DEL15, CANCELLED, DIVERTED, AIR_TIME
# MAGIC FROM airlines_vw 
# MAGIC WHERE TAIL_NUM in ('N658MQ', 'N688MQ', 'N621MQ', 'N988CA', 'N963SW', 'N952SW', 'N16911', 'N3LTAA', 'N838UA', 'N637JB', 'N87527', 'N37298', 'N645MQ')
# MAGIC   AND FL_DATE >= '2015-02-01' and FL_DATE < '2015-03-01'
# MAGIC   ORDER By TAIL_NUM, FL_DATE

# COMMAND ----------

# MAGIC %md ## Data Issues

# COMMAND ----------

# MAGIC %md
# MAGIC * Null values in TAIL_NUM
# MAGIC * The arrival and departure times are strings and local times. Will need some work for us to correlate the weather data.
# MAGIC * FL_DATE field has the date (local time based) of departure. Depending on the departure time, the arrival date could be different. For example, for a flight departing at say 23:30 having an arrival time of 630, we need to compute the date appropriately. Need to evaluate if its safe to assume that the arrival date will never be more then departure date + 1 day.
# MAGIC * `DEP_TIME_BLK` and `ARR_TIME_BLK` might be useful. They identify the hour of the day. All flights arriving in the same hour have the same value in `ARR_TIME_BLK`. All flights departing in the same hour have the same value in `DEP_TIME_BLK`. 
# MAGIC * There are 8 airports missing corresponding weather station - `PSE, PPG, OGS, SPN, SJU, TKI, GUM, XWA`. Of these, `SJU` and `GUM` are large airports.
# MAGIC 
# MAGIC  

# COMMAND ----------

# MAGIC %md ### Null Values Exploration

# COMMAND ----------

# MAGIC %md ### Neighboring WeatherStations

# COMMAND ----------

import pandas as pd
import geopandas
import matplotlib.pyplot as plt
def try_gpd():
    df = pd.DataFrame(
    {'Airport': ['SFO', 'SJC'],
     'Latitude': [37.362598, 37.61899948120117],
     'Longitude': [-121.929001, -122.375]})
    gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    display(world)
    # We restrict to South America.
    ax = world[world.iso_a3 == 'USA'].plot(
        edgecolor='black')

    # We can now plot our ``GeoDataFrame``.
    gdf.plot(ax=ax, color='red')

    plt.show()
try_gpd()
    

# COMMAND ----------

# MAGIC %md ### Weather Data Exploration

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct station) from weather_vw_src

# COMMAND ----------

def test():
#   spark.conf.set("spark.sql.parquet.enableVectorizedReader","false")
  df = spark.read.parquet(WEATHER_RAW_LOC)
  display(df.filter(df.STATION == '72657594960'))

test()

# COMMAND ----------

# MAGIC %sh du -h /dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp

# COMMAND ----------

# MAGIC %sh du -h /dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2015

# COMMAND ----------

# MAGIC %md # Check Weather Station Counts

# COMMAND ----------

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

def load_views_for_weather_station_analysis():
    spark.read.parquet(AIRPORTS_MASTER_LOC).createOrReplaceTempView("vw_airports_master")
    spark.read.parquet(AIRPORTS_WS_LOC).createOrReplaceTempView("vw_airports_ws")
    spark.read.parquet(f'{WEATHER_LOC}/clean_weather_data.parquet').createOrReplaceTempView("vw_weather_cleaned")
    spark.read.parquet(FLIGHT_PRE_COVID_RAW_LOC).createOrReplaceTempView("vw_flight_raw")
    spark.read.parquet(FINAL_JOINED_DATA_ALL).createOrReplaceTempView("vw_final_joined")
    spark.read.parquet(WEATHER_RAW_LOC).createOrReplaceTempView("vw_weather_raw")
    spark.read.parquet(WEATHER_STATIONS_RAW_LOC).createOrReplaceTempView("vw_ws_raw")
load_views_for_weather_station_analysis()


# COMMAND ----------

# MAGIC %sql
# MAGIC WITH origin_airport as (
# MAGIC   SELECT origin_airport_iata, origin_airport_ws_station_id
# MAGIC   FROM vw_final_joined
# MAGIC ), dest_airport as (
# MAGIC   SELECT dest_airport_iata, dest_airport_ws_station_id
# MAGIC   FROM vw_final_joined
# MAGIC ), weather_stations as (
# MAGIC   SELECT origin_airport_ws_station_id as weather_station_id FROM origin_airport
# MAGIC   UNION SELECT dest_airport_ws_station_id as weather_station_id FROM dest_airport
# MAGIC )
# MAGIC SELECT distinct weather_station_id FROM weather_stations 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT distinct ws_station_id from vw_airports_ws

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct STATION from vw_weather_cleaned

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH 
# MAGIC   stations_airports_ws as (SELECT distinct ws_station_id as station_id from vw_airports_ws),
# MAGIC   weather_raw as (select distinct STATION as station_id from vw_weather_raw),
# MAGIC   weather_cleaned as (select distinct STATION as station_id from vw_weather_cleaned)
# MAGIC SELECT 
# MAGIC     a.station_id as airports_ws_id, b.station_id as weather_raw_ws_id,
# MAGIC     c.station_id as weather_cleaned_ws_id 
# MAGIC   FROM stations_airports_ws a
# MAGIC   FULL OUTER JOIN weather_raw as b on a.station_id = b.station_id
# MAGIC   FULL OUTER JOIN weather_cleaned as c on a.station_id = c.station_id
# MAGIC   WHERE a.station_id is null or b.station_id is null or c.station_id is null
# MAGIC   ORDER BY c.station_id, b.station_id

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH 
# MAGIC   weather_stations as (
# MAGIC     SELECT origin_airport_iata as iata, origin_airport_ws_station_id as weather_station_id
# MAGIC     FROM vw_final_joined
# MAGIC     UNION
# MAGIC     SELECT dest_airport_iata as iata, dest_airport_ws_station_id as weather_station_id
# MAGIC     FROM vw_final_joined
# MAGIC   )
# MAGIC SELECT distinct iata, weather_station_id FROM weather_stations 
# MAGIC WHERE weather_station_id in ('72314013881')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT airports.iata, stations.* from vw_ws_raw stations
# MAGIC JOIN vw_airports_master airports ON stations.neighbor_call = airports.icao
# MAGIC WHERE station_id in ('72208613758', '72297903157', '72312693804', '72389723167', '70027127506', '70063627405', '72205112841', '69014093101', '72530614855', '72534694866', '72676324198', '74506023239', '91178022514', 'A0002253995', '72406593733', '72224503882', '72407613724', '69007093217', '72408614793', '72258593901', '72456713921', '72469023062', '72267523021', '72509714790', '72512014761', '72280023195', '72512114761', '72513514754', '72291593114', '72518794733', '70316025624', '72064354940', '72255012912', '72445703938', '72451013985', '72462023061', '72480023157', '72512704726', '72514014778', '72531603887', '72547804920', '72548514940', '72549094933', '72666024029', '72672024061', '72781024243', '72782594239', '72784624160', '72785794129')
# MAGIC AND neighbor_name like '%AIRPORT%' and distance_to_neighbor < 20
# MAGIC order by station_id

# COMMAND ----------

def load_views():
    spark.read.parquet(AIRPORTS_MASTER_LOC).createOrReplaceTempView("vw_airports_master")
    spark.read.parquet(AIRPORTS_WS_LOC).createOrReplaceTempView("vw_airports_ws")
    spark.read.parquet(FLIGHT_PRE_COVID_RAW_LOC).createOrReplaceTempView("vw_pre_covid_flights")
    spark.read.parquet(FLIGHT_COVID_RAW_LOC).createOrReplaceTempView("vw_covid_flights")
    
def to_map(schema):
    columns = {}
    for col in schema:
        columns[col.name] = col.dataType
    return columns

def to_set(schema):
    columns = set()
    for col in schema:
        columns.add(f'{col.name.upper().replace("_", "")}::{col.dataType}')
    return columns

def check_columns():
    df_pre_covid_flights = spark.sql("SELECT * from vw_pre_covid_flights")
#     pre_covid_cols = to_map(df_pre_covid_flights.schema)
#     pre_covid_set = to_set(df_pre_covid_flights.schema)

    df_covid_flights = spark.sql("SELECT * from vw_covid_flights")
    
    print('---------------')
    print(df_pre_covid_flights.schema)
    print('---------------')
    print(df_covid_flights.schema)
    print('---------------')

#     covid_set = to_set(df_covid_flights.schema)
#     covid_cols = to_map(df_covid_flights.schema)    
#     print(pre_covid_cols)
#     print(covid_cols)
#     print('---------------')
#     print(pre_covid_set ^ covid_set)

#     print('---------------')
#     print(pre_covid_set)
#     print('---------------')
#     print(covid_set)
#     print('---------------')
    
load_views()
check_columns()

# COMMAND ----------

def test():
    training = spark.read.parquet(FINAL_JOINED_DATA_TRAINING)
    testing = spark.read.parquet(FINAL_JOINED_DATA_TEST)
    
    print(training.count())
    print(testing.count())
    
    print(training.columns)
test()

# COMMAND ----------


