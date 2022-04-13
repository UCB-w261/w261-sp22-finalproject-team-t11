# Databricks notebook source
# MAGIC %md # Data ETL
# MAGIC This notebook has the steps and code for preparing data for analysis and machine learning.

# COMMAND ----------

# MAGIC %md ## Notebook Setup
# MAGIC Import necessary libraries, setup up authentication for reading from and writing to the blob store.

# COMMAND ----------

import os
from pyspark import SparkFiles
from pyspark.sql.functions import col, when, to_utc_timestamp, to_timestamp, year, date_trunc, split, regexp_replace, array_max, length, substring, greatest, minute, hour, expr, count
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import geopandas as gpd
import seaborn as sns


%matplotlib inline

# COMMAND ----------

from configuration_v01 import Configuration
configuration = Configuration()

# Define constants
# Location of raw data.
IATA_TZ_MAP_RAW_LOC = f"{configuration.blob_url}/raw/iata_tz_map"
AIRPORTS_CODE_RAW_LOC = f"{configuration.blob_url}/raw/airport_codes"
AIRPORTS_MASTER_RAW_LOC = f"{configuration.blob_url}/raw/airports"

# Original given sources:
FLIGHT_PRE_COVID_RAW_LOC =  "/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*"
WEATHER_STATIONS_RAW_LOC = "/mnt/mids-w261/datasets_final_project/stations_data/*"
WEATHER_RAW_LOC = "/mnt/mids-w261/datasets_final_project/weather_data/*"

# New data sources:
FLIGHT_COVID_RAW_LOC = f"{configuration.blob_url}/raw/flights_covid"
# WEATHER_STATIONS_RAW_LOC = f"{blob_url}/raw/stations"
WEATHER_RAW_LOC = f"{configuration.blob_url}/raw/weather"

# Location of staged data.
# AIRPORT_WEATHER_LOC = f"{configuration.blob_url}/raw/airport_weather"
AIRPORTS_MASTER_LOC = f"{configuration.blob_url}/staged/airports"
AIRPORTS_WS_LOC = f"{configuration.blob_url}/staged/airports_weatherstations"
WEATHER_LOC = f"{configuration.blob_url}/staged/weather"
CLEAN_WEATHER_LOC = f'{WEATHER_LOC}/clean_weather_data.parquet'

# 3 months of data for testing and validating code
FINAL_JOINED_DATA_2015_Q1 = f"{configuration.blob_url}/staged/final_joined_2015_Q1"
FINAL_JOINED_DATA_Q1_2019 = f"{configuration.blob_url}/staged/final_joined_Q1_2019"


# COMMAND ----------

# MAGIC %md ## Upload Airport Timezones Data
# MAGIC 
# MAGIC Source: http://www.fresse.org/dateutils/tzmaps.html

# COMMAND ----------

def copy_airport_tz_to_blob_store():
    df_delay = spark.read.csv("/user/ram.senth@berkeley.edu/iata_tzmap.csv", header=True, inferSchema= True)
    df_delay.write.mode('overwrite').parquet(f"{configuration.blob_url}/raw/iata_tz_map")
# Need to run this just once.
# copy_airport_tz_to_blob_store()

# COMMAND ----------

# MAGIC %md ## Upload Airport Codes
# MAGIC 
# MAGIC Source: https://datahub.io/core/airport-codes#resource-airport-codes

# COMMAND ----------

def copy_airport_codes_to_blob_store():
    df_delay = spark.read.csv("/user/ram.senth@berkeley.edu/airport-codes.csv", header=True, inferSchema= True)
    df_delay.write.mode('overwrite').parquet(f"{configuration.blob_url}/raw/airport_codes")
# Need to run this just once.
# copy_airport_codes_to_blob_store()

# COMMAND ----------

# MAGIC %md ## Load Source Data
# MAGIC Loads the following datasets from the specified locations:
# MAGIC 
# MAGIC |Dataset|Location|
# MAGIC |-------|--------|
# MAGIC |Airport Codes|{blob_url}/raw/airport_codes|
# MAGIC |Airport Timezones|{blob_url}/raw/iata_tz_map|
# MAGIC |Weather|/mnt/mids-w261/datasets_final_project/stations_data|
# MAGIC |Flights|/mnt/mids-w261/datasets_final_project/parquet_airlines_data|
# MAGIC |Weather Stations|/mnt/mids-w261/datasets_final_project/weather_data|

# COMMAND ----------

# MAGIC %md ## Create Airports Master
# MAGIC Our airports master will have the following fields and have records for all airports found in the full flights data set (both covid and pre-covid):
# MAGIC 
# MAGIC | Field | Type | Description | Source |
# MAGIC |-------|------|-------------|--------|
# MAGIC | name | String | Name of the airport | Flights |
# MAGIC | iata | String| 3 letter IATA code | Flights |
# MAGIC | icao | String| 4 letter ICAO code | Airport Codes |
# MAGIC | tz | String | IANA timezone name | Airport Timezones |
# MAGIC | type | Number | One of "Small Airport", "Medium Airport" and "Large Airport". | Airport Codes |
# MAGIC | elevation| Number | Airport Elevation (in feet) | Airport Codes |
# MAGIC | airport_lat | Number | The airport latitude in decimal degrees (positive for north). | Airport Codes |
# MAGIC | airport_lon | Number | The airport longitude in decimal degrees (positive for east). |
# MAGIC | iso_country | String | The ISO country code for location of the airport. | Airport Codes |
# MAGIC | iso_region | String | The ISO region code for location of the airport. | Airport Codes |
# MAGIC | municipality | String | Location of the airport. | Airport Codes | Airport Codes |
# MAGIC | ws_station_id | Number | Primary weather station id associated with the airport. Null if one not defined. | Weather Stations |
# MAGIC | ws_lat | Number | Lattitude for the weather station location. | Weather Stations |
# MAGIC | ws_lon | Number | Longitude for the weather station location. | Weather Stations |
# MAGIC 
# MAGIC We see that the following airports do not have an associated weather station:
# MAGIC |Airports Missing Weather Station |IATA|
# MAGIC |----|----|
# MAGIC | Pago Pago International Airport | PPG |
# MAGIC | Ogdensburg International Airport | OGS |
# MAGIC | Saipan International Airport | SPN |
# MAGIC | Tokeen Seaplane Base | TKI |
# MAGIC | Antonio B. Won Pat International Airport | GUM |
# MAGIC 
# MAGIC 
# MAGIC Flights in and out of these airports (between 2015 and 2019):
# MAGIC | Airport | Outbound | Inbound |
# MAGIC |---------|--------|--------|
# MAGIC | PSE | 8062 | 8088 |
# MAGIC | GUM | 5118 | 5118 |
# MAGIC | SPN | 1462 | 1462 |
# MAGIC | OGS | 1332 | 1334 |
# MAGIC | PPG | 1196 | 1196 |
# MAGIC | TKI | 2 | null |
# MAGIC 
# MAGIC Since these airports do not see high volume of flights, we will drop all flights from and to these airports from our dataset.
# MAGIC 
# MAGIC ### Data Audit
# MAGIC Total airports (covid + pre-covid): 389
# MAGIC 
# MAGIC Airports missing weather station: 6

# COMMAND ----------

def load_data_for_airports_master():
    df_pre_covid_flights = spark.read.parquet(FLIGHT_PRE_COVID_RAW_LOC)
    df_pre_covid_flights.createOrReplaceTempView("vw_pre_covid_flights")

    df_covid_flights = spark.read.parquet(FLIGHT_COVID_RAW_LOC)
    df_covid_flights.createOrReplaceTempView("vw_covid_flights")

    df_airport_codes_raw = spark.read.parquet(AIRPORTS_CODE_RAW_LOC)
    df_airport_codes_raw.createOrReplaceTempView("vw_airport_codes_raw")

    df_stations_src = spark.read.parquet(WEATHER_STATIONS_RAW_LOC)
    df_stations_src.createOrReplaceTempView("vw_stations_src")

def iatas_in_dataset():
    sql = """
        with iatas_pre_covid as (
            SELECT DISTINCT ORIGIN as iata FROM vw_pre_covid_flights UNION SELECT DISTINCT DEST as iata FROM vw_pre_covid_flights
        ),
        iatas_covid as (
            SELECT DISTINCT ORIGIN as iata FROM vw_covid_flights UNION SELECT DISTINCT DEST as iata FROM vw_covid_flights
        ),
        all_iatas as (SELECT DISTINCT iata FROM iatas_pre_covid UNION SELECT DISTINCT iata FROM iatas_covid)
        SELECT distinct iata from all_iatas ORDER by iata
    """
    df_iatas = spark.sql(sql)
    df_iatas.cache()
    df_iatas.createOrReplaceTempView("vw_iatas")
    print(f'Total number of airports of interest: {df_iatas.count()}')

load_data_for_airports_master()
iatas_in_dataset()    


# COMMAND ----------

def create_airports():
    sql = """ WITH
          iatas_of_interest as (SELECT iata FROM vw_iatas),
          airport_codes as (
            SELECT type, name, iso_country, iso_region, gps_code as icao, iata_code as iata, 
                local_code, municipality, coordinates, elevation_ft as elevation,
                cast(split(coordinates, ',')[0] as double) as airport_lon, 
                cast(split(coordinates, ',')[1] as double) as airport_lat
              FROM vw_airport_codes_raw
          ),
          stations as (
            SELECT station_id as ws_station_id, neighbor_id, neighbor_call, neighbor_name, distance_to_neighbor, 
                cast(lat as double) as ws_lat, cast(lon as double) as ws_lon
              FROM vw_stations_src
          ),
          airports as (
            SELECT ac.name, ioi.iata, ac.icao, tzs.iana_tz_name as tz, ac.type, 
                ac.elevation as elevation, airport_lat, airport_lon, 
                ac.iso_country, ac.iso_region, ac.municipality, 
                s.ws_station_id, s.distance_to_neighbor as ws_distance, 
                s.ws_lat, s.ws_lon
              FROM iatas_of_interest ioi 
              LEFT OUTER JOIN airport_codes ac ON ioi.iata = ac.iata
              LEFT OUTER JOIN stations s ON ac.icao = s.neighbor_call and s.distance_to_neighbor < 1
              LEFT OUTER JOIN vw_airport_tzs_raw tzs on ioi.iata = tzs.iata_code
          )
        SELECT * FROM airports
    """
    df_airports_src = spark.sql(sql)
    df_airports_src.coalesce(1).write.mode('overwrite').parquet(AIRPORTS_MASTER_RAW_LOC)
    print(f'Total number of airport master records created: {df_airports_src.count()}')
    
create_airports()
    


# COMMAND ----------

def airports_missing_weatherstations():
    df_airports_raw = spark.read.parquet(AIRPORTS_MASTER_RAW_LOC)
    display(df_airports_raw.select('name', 'iata').where('ws_station_id is null'))

airports_missing_weatherstations()
    

# COMMAND ----------

def flights_from_these_airports(flights_table):
    base_sql = """
        WITH 
            forward as (
              SELECT ORIGIN as primary, DEST as secondary, count(*) as counts
                FROM """ + flights_table + """
                WHERE ORIGIN in ('PSE', 'PPG', 'OGS', 'SPN', 'SJU', 'TKI', 'GUM', 'XWA')
                GROUP BY ORIGIN, DEST
            ),
            reverse as (
              SELECT DEST as primary, ORIGIN as secondary, count(*) as counts
                FROM """ + flights_table + """
                WHERE DEST in ('PSE', 'PPG', 'OGS', 'SPN', 'SJU', 'TKI', 'GUM', 'XWA')
                GROUP BY DEST, ORIGIN
            ) 
    """
    sql = base_sql + """
        SELECT f.primary, f.secondary, f.counts as onward, r.counts as return
            FROM forward f
            FULL OUTER JOIN reverse r 
                ON f.primary = r.primary and f.secondary = r.secondary
            ORDER by f.counts desc
    """
    df_flights = spark.sql(sql)
    display(df_flights)

    sql = base_sql + """
        SELECT f.primary, sum(f.counts) as onward, sum(r.counts) as return
            FROM forward f
            FULL OUTER JOIN reverse r 
                ON f.primary = r.primary and f.secondary = r.secondary
            group by f.primary
            ORDER by onward desc
    """
    df_flights = spark.sql(sql)
    display(df_flights)
    
flights_from_these_airports('vw_pre_covid_flights')
flights_from_these_airports('vw_covid_flights')

# COMMAND ----------

# MAGIC %md ### Weather Stations Closest To Airports

# COMMAND ----------

def load_airports_master_raw():
    # Load the previously created airports master.
    df_airports_raw = spark.read.parquet(AIRPORTS_MASTER_RAW_LOC)
    df_airports_raw.createOrReplaceTempView("vw_airports_master_raw")

load_airports_master_raw()

# COMMAND ----------

def neighbor_search(max_miles, k):
    df_airport_locs = spark.sql("""
        SELECT iata, COALESCE(ws_lat, airport_lat) as airport_lat, COALESCE(ws_lon, airport_lon) as airport_lon 
        FROM vw_airports_master_raw"""
    ).toPandas()

    df_ws_locs = spark.sql("""
        SELECT DISTINCT station_id as ws_station_id, lat as ws_lat, lon as ws_lon FROM vw_stations_src"""
    ).toPandas()

    # Setup Balltree using df_airport_locs as reference dataset
    # Use Haversine calculate distance between points on the earth from lat/long
    # haversine - https://pypi.org/project/haversine/ 
    tree = BallTree(np.deg2rad(df_ws_locs[['ws_lat', 'ws_lon']].values), metric='haversine')

    airport_lats = df_airport_locs['airport_lat']
    airport_lons = df_airport_locs['airport_lon']

    distances, indices = tree.query(np.deg2rad(np.c_[airport_lats, airport_lons]), k)
    
    airports_ws = []
    r_km = 6371 # multiplier to convert to km (from unit returned by haversine for lattitude/lomgitude)
    r_miles = 3958.756 # multiplier to convert to miles (from unit returned by haversine for lattitude/lomgitude)
    station_ids = df_ws_locs['ws_station_id']
    for iata, d, ind in zip(df_airport_locs['iata'], distances, indices):
        for i, index in enumerate(ind):
            miles = round(d[i]*r_miles, 4)
            if miles > max_miles:
                break
            ws_lat = df_ws_locs['ws_lat'][index]
            ws_lon = df_ws_locs['ws_lon'][index]
            airports_ws.append([iata, df_ws_locs['ws_station_id'][index], ws_lat, ws_lon, miles])
    return spark.createDataFrame(pd.DataFrame(airports_ws, columns = ['iata', 'ws_station_id', 'ws_lat', 'ws_lon', 'miles']))

def create_airports_ws():
    df = neighbor_search(max_miles=20, k=5)
    #Save the data
    df.coalesce(1).write.mode('overwrite').parquet(AIRPORTS_WS_LOC)

    # print dataframe.
    display(df)

create_airports_ws()


# COMMAND ----------

def test():
    df = spark.read.parquet(AIRPORTS_WS_LOC)
    df.createOrReplaceTempView("vw_airports_ws")
    display(spark.sql("SELECT iata, count(*) FROM vw_airports_ws group by iata"))
    print(df.count())
    display(df.filter(col("iata").isin('PSE', 'PPG', 'OGS', 'SPN', 'SJU', 'TKI', 'GUM', 'XWA', 'SFO')))

test()

# COMMAND ----------

def plot_many(data):
    # Setup temp view for the airports master data.
    df_airports = spark.read.parquet(AIRPORTS_MASTER_RAW_LOC)
    df_airports.createOrReplaceTempView("vw_airports_master_raw")

    num_plots = len(data)
    
    fig, axis = plt.subplots(1, num_plots, figsize=(12, 12), tight_layout=True)

    for i, (state_name, iatas, shapefile, crs) in enumerate(data):
        ax = axis[i]
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{state_name}')
        plot(ax, i, iatas, shapefile, crs)
        
    fig.suptitle('Upto 5 Closest Weather Stations Within 20 Miles', fontsize=16)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.tight_layout()
    plt.show()

        
def plot(ax, index, iatas, shapefile, crs):
    # Load the shape file for the state.
    state = gpd.read_file(f'{configuration.SHAPES_BASE_FOLDER}/{shapefile}')
    
    # Set scaling if needed.
    if crs is None:
        state.plot(ax=ax)
    else:
        state.to_crs(epsg=crs).plot(ax=ax)
        
    iatas_str = "'" + "','".join(iatas) + "'"
    
    # Load the airports data and plot it.
    df_airport_locs = spark.sql(f"""
        SELECT iata, name, airport_lat, airport_lon
        FROM vw_airports_master_raw WHERE iata in ({iatas_str})"""
    ).toPandas()
    gdf_airports = gpd.GeoDataFrame(df_airport_locs, geometry=gpd.points_from_xy(df_airport_locs.airport_lon, df_airport_locs.airport_lat))
    gdf_airports.plot(ax=ax, color='red', label=df_airport_locs['iata'])
    for ind in gdf_airports.index:
        ax.text(df_airport_locs['airport_lon'][ind], df_airport_locs['airport_lat'][ind], s=df_airport_locs['name'][ind], horizontalalignment='left', verticalalignment='top', bbox={'facecolor': 'white', 'alpha':0.2, 'pad': 3, 'edgecolor':'none'})
    
    # County filtering logic, if we want to disply partial state.
#     counties = {'LAKE', 'COOK', 'DUPAGE', 'WILL', 'GRUNDY', 'KENDALL', 'KANE', 'MCHENRY'}
#     criterion = state['COUNTY_NAM'].map(lambda x: x in counties)
#     state[criterion].plot(ax=ax)
    
    # Load and plot weather stations data
    df_stations_spark = spark.read.parquet(AIRPORTS_WS_LOC)
    df_stations = df_stations_spark.filter(f"iata in ({iatas_str})").toPandas()
    gdf_ws = gpd.GeoDataFrame(df_stations, geometry=gpd.points_from_xy(df_stations.ws_lon, df_stations.ws_lat))
    gdf_ws.plot(ax=ax, color='lightgreen', alpha=0.5, legend=True)
    

plot_many([('Illinois', ['ORD'], 'illinois_counties/IL_BNDY_County_Py.shp', None), 
           ('Georgia', ['ATL'], 'georgia_counties/GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp', None)])

# plot_many([(['SFO', 'LAX'], '/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/shapes/ca_counties/ca_counties.shp', 4326)])



# COMMAND ----------

def remove_airports_missing_weatherstations():
    df_airports_raw = spark.read.parquet(AIRPORTS_MASTER_RAW_LOC)
    df_airports_raw.createOrReplaceTempView("vw_airports_master_raw")
    df_airports_ws = spark.read.parquet(AIRPORTS_WS_LOC)
    df_airports_ws.createOrReplaceTempView("vw_airports_ws")
    sql = """
        WITH airports_to_ignore as (
            SELECT DISTINCT am.iata as iata
                FROM vw_airports_master_raw as am
                LEFT OUTER JOIN vw_airports_ws as aw ON am.iata = aw.iata
                WHERE aw.iata is null
        )
        SELECT am.* 
            FROM vw_airports_master_raw as am
            WHERE am.iata not in (SELECT iata from airports_to_ignore)
    """
    df_final_airports = spark.sql(sql)
#     df_final_airports.coalesce(1).write.mode('overwrite').parquet(AIRPORTS_MASTER_LOC)
#     df_airports = spark.read.parquet(AIRPORTS_MASTER_LOC)
#     df_airports.createOrReplaceTempView("vw_airports_master")
    df_final_airports.createOrReplaceTempView("vw_airports_master")
    
def test():
    display(spark.sql('SELECT count(*) FROM vw_airports_master'))

remove_airports_missing_weatherstations()
test()



# COMMAND ----------

# MAGIC %md ### Weather Stations Closer To Multiple Airports
# MAGIC 
# MAGIC We find that there are about 50 weather stations that are associated with more than 1 airport and in one case (water station id 72767094014, shared between airports ISN and XWA) there are two different timezones. We need to aggregate weather at airport level and store in local timezone for airport.

# COMMAND ----------

def load_data_for_ws_closer_to_many_airports():
    df_airports_ws = spark.read.parquet(AIRPORTS_WS_LOC)
    df_airports_ws.createOrReplaceTempView("vw_airports_ws")

def find_ws_closer_to_many_airports():
    load_data_for_ws_closer_to_many_airports()
    sql = """
        WITH ws_with_many_airports as (
            SELECT ws_station_id, count(*) as counts from vw_airports_ws
            GROUP BY ws_station_id
            HAVING counts > 1),
            iatas as (
                SELECT iata, ws_station_id from vw_airports_ws 
                    WHERE ws_station_id in (SELECT ws_station_id FROM ws_with_many_airports))
        SELECT iatas.ws_station_id, am.* 
            FROM vw_airports_master am 
            JOIN iatas ON am.iata = iatas.iata
            ORDER BY iatas.ws_station_id
    """
    display(spark.sql(sql))
    
find_ws_closer_to_many_airports()

# COMMAND ----------

# MAGIC %md ### Update Closest Weather Station for Airports Missing WS

# COMMAND ----------

def update():
    # Find the nearest weather stations for these airports.
    spark.read.parquet(AIRPORTS_WS_LOC).createOrReplaceTempView("vw_airports_ws")
    sql = """
        SELECT a.name, a.iata, a.icao, a.tz, a.type, a.elevation, a.airport_lat, a.airport_lon, a.iso_country, a.iso_region, a.municipality, 
            b.ws_station_id as ws_station_id, 
            b.miles as ws_distance,
            b.ws_lat as ws_lat,
            b.ws_lon as ws_lon
            FROM vw_airports_master a 
            JOIN vw_airports_ws b on a.iata = b.iata
            WHERE a.ws_station_id is null
        UNION
        SELECT a.name, a.iata, a.icao, a.tz, a.type, a.elevation, a.airport_lat, a.airport_lon, a.iso_country, a.iso_region, a.municipality, 
            a.ws_station_id as ws_station_id, 
            a.ws_distance as ws_distance,
            a.ws_lat as ws_lat,
            a.ws_lon as ws_lon
            FROM vw_airports_master a 
            JOIN vw_airports_ws b on a.iata = b.iata
            WHERE a.ws_station_id is not null        
    """
    df_airports_final = spark.sql(sql)
    df_airports_final.coalesce(1).write.mode('overwrite').parquet(AIRPORTS_MASTER_LOC)
    df_airports = spark.read.parquet(AIRPORTS_MASTER_LOC)
    df_airports.createOrReplaceTempView("vw_airports_master")

update()

# COMMAND ----------

# MAGIC %md ## Stage Weather Data
# MAGIC 
# MAGIC In this approach we adopt the simple strategy of averaging the values of interest from the weather table. We also do some data clean-up, keeping only the columns of interest.

# COMMAND ----------

def load_and_transform_data():
    """
        Load selected columns from weather data, transform the data for correct data types, split up complex columns into individual fields.
        Also create the needed viewes for the final query.
    """
    drop_cols = ('HourlySkyConditions', 'HourlyPresentWeatherType', 'HourlyPrecipitation', 'HourlyPressureTendency')
    cols_to_keep = ['STATION',
                    'DATE',
                    'LATITUDE', 'LONGITUDE', 'ELEVATION',
                    'NAME',
                    'REPORT_TYPE',
                    'SOURCE',
                    'HourlyAltimeterSetting',
                    'HourlyDewPointTemperature', 'HourlyDryBulbTemperature',
                    'HourlyPrecipitation',
                    'HourlyPresentWeatherType',
                    'HourlyPressureChange', 'HourlyPressureTendency',
                    'HourlyRelativeHumidity',
                    'HourlySkyConditions',
                    'HourlySeaLevelPressure', 'HourlyStationPressure',
                    'HourlyVisibility', 'HourlyWetBulbTemperature',
                    'HourlyWindDirection', 'HourlyWindGustSpeed', 'HourlyWindSpeed']

    df_weather = spark.read.parquet(f"{blob_url}/raw/weather")
    df_weather_selected = df_weather.select(*cols_to_keep)
    display(df_weather_selected)
    df_weather_selected.createOrReplaceTempView("vw_weather_selected")
    
    # Convert DATE Column from string to timestamp
    df_weather_selected = df_weather_selected.withColumn("DATE", df_weather_selected['DATE'].cast('timestamp'))
    
    # The actual impact of each weather record is only in the next hour as we aggregate by the hour for past data.
    # This takes care of data leakage issue.
    df_weather_selected = df_weather_selected.withColumn('DATE_IMPACT', df_weather_selected.DATE + expr('INTERVAL 1 HOURS') -  expr('INTERVAL 1 MINUTES'))

    # Create new column with the hour of the weather reading (get rid of the minute reading)
    df_weather_selected = df_weather_selected.withColumn('DATEHOUR', date_trunc("hour", df_weather_selected.DATE + expr('INTERVAL 1 HOURS') -  expr('INTERVAL 1 MINUTES')))

    # Convert Latitude, Longitude, Elevation from string to double
    df_weather_selected = df_weather_selected.withColumn("LATITUDE",  df_weather_selected['LATITUDE'].cast('double'))
    df_weather_selected = df_weather_selected.withColumn("LONGITUDE",  df_weather_selected['LONGITUDE'].cast('double'))
    df_weather_selected = df_weather_selected.withColumn("ELEVATION",  df_weather_selected['ELEVATION'].cast('double'))

    # Apply transformations to convert weather conditions to appropriate values

    # HOURLY ALTIMETER SETTING
    df_weather_selected = df_weather_selected.withColumn('HourlyAltimeterSetting', 
                                                                   split(df_weather_selected["HourlyAltimeterSetting"],'s')[0]\
                                                                         .cast('double'))

    # DEW POINT TEMPERATURE
    df_weather_selected = df_weather_selected.withColumn('HourlyDewPointTemperature', 
                           split(df_weather_selected["HourlyDewPointTemperature"],'s')[0])
    df_weather_selected = df_weather_selected.withColumn('HourlyDewPointTemperature',
                           df_weather_selected["HourlyDewPointTemperature"].cast('double'))

    # DRY BULB TEMPERATURE
    df_weather_selected = df_weather_selected.withColumn('HourlyDryBulbTemperature', 
                           split(df_weather_selected["HourlyDryBulbTemperature"],'s')[0])
    df_weather_selected = df_weather_selected.withColumn('HourlyDryBulbTemperature',
                           df_weather_selected["HourlyDryBulbTemperature"].cast('double'))

    ## Process PRECIPITATION
    df_weather_selected = df_weather_selected.withColumn('Precip_Double', 
                                                                   split(df_weather_selected["HourlyPrecipitation"], 
                                                                         's')[0]) # get rid of s
    df_weather_selected = df_weather_selected.withColumn('Precip_Double', 
                       regexp_replace('Precip_Double', 'T', '0')) # replace T with 0
    df_weather_selected = df_weather_selected.withColumn('Precip_Double', 
                       regexp_replace('Precip_Double', '\*', '0')) # replace * with 0

    # if there are multiple values in a single cell, get the maximum, cast the value as a double, replace NA values with 0
    df_weather_selected = df_weather_selected.withColumn('Precip_Double', 
                                                                   when(length(df_weather_selected['Precip_Double']) % 4 == 0, 
                                                                        greatest(substring('Precip_Double', 1, 4), 
                                                                                 substring('Precip_Double', 5, 4),
                                                                                 substring('Precip_Double', 9, 4)))\
                                                                   .when(length(df_weather_selected['Precip_Double']) % 5 == 0,
                                                                         greatest(substring('Precip_Double', 1, 5), 
                                                                                  substring('Precip_Double', 6, 5),
                                                                                  substring('Precip_Double', 11, 5)))\
                                                                   .otherwise(df_weather_selected['Precip_Double'])\
                                .cast('double'))\
                                .na.fill(0.0, subset = ['Precip_Double'])
    df_weather_selected = df_weather_selected.withColumn('Trace_Rain', 
                                                                   when(df_weather_selected['HourlyPrecipitation'].contains('T'), 
                                                                        1).otherwise(0))
    df_weather_selected = df_weather_selected.withColumn('NonZero_Rain', 
                                                                   when((df_weather_selected['Trace_Rain'] == 1) | \
                                                                        (df_weather_selected['Precip_Double'] > 0), 
                                                                        1).otherwise(0))

    # HOURLY PRESSURE CHANGE
    df_weather_selected = df_weather_selected.withColumn('HourlyPressureChange', 
                                                                   df_weather_selected['HourlyPressureChange'].cast('double'))

    # HOURLY PRESSURE TENDENCY
    # CONVERT TO 3 SEPARATE BINARY COLUMNS BASED ON CATEGORICAL VALUE
    df_weather_selected = df_weather_selected.withColumn('HourlyPressureTendency', 
                                                                   df_weather_selected['HourlyPressureChange'].cast('double'))

    df_weather_selected = df_weather_selected.withColumn('HourlyPressureTendency_Increasing',
                                                                   when(df_weather_selected['HourlyPressureTendency'] <= 3,
                                                                        1)\
                                                                   .otherwise(0))
    df_weather_selected = df_weather_selected.withColumn('HourlyPressureTendency_Decreasing',
                                                                   when(df_weather_selected['HourlyPressureTendency'] >= 5,
                                                                        1)\
                                                                   .otherwise(0))
    df_weather_selected = df_weather_selected.withColumn('HourlyPressureTendency_Constant',
                                                                   when(df_weather_selected['HourlyPressureTendency'] == 4,
                                                                        1)\
                                                                   .otherwise(0))

    # HOURLY RELATIVE HUMIDITY
    df_weather_selected = df_weather_selected.withColumn('HourlyRelativeHumidity', 
                           split(df_weather_selected["HourlyRelativeHumidity"],'s')[0])
    df_weather_selected = df_weather_selected.withColumn('HourlyRelativeHumidity',
                           df_weather_selected["HourlyRelativeHumidity"].cast('double'))

    # HOURLY SEA LEVEL PRESSURE
    df_weather_selected = df_weather_selected.withColumn('HourlySeaLevelPressure', 
                           split(df_weather_selected["HourlySeaLevelPressure"],'s')[0])
    df_weather_selected = df_weather_selected.withColumn('HourlySeaLevelPressure',
                           df_weather_selected["HourlySeaLevelPressure"].cast('double'))

    # HOURLY STATION PRESSURE
    df_weather_selected = df_weather_selected.withColumn('HourlyStationPressure', 
                           split(df_weather_selected["HourlyStationPressure"],'s')[0])
    df_weather_selected = df_weather_selected.withColumn('HourlyStationPressure',
                           df_weather_selected["HourlyStationPressure"].cast('double'))

    # HOURLY HORIZONTAL VISIBILITY
    df_weather_selected = df_weather_selected.withColumn('HourlyVisibility', 
                           split(df_weather_selected["HourlyVisibility"],'s')[0])
    df_weather_selected = df_weather_selected.withColumn('HourlyVisibility', 
                           split(df_weather_selected["HourlyVisibility"],'V')[0])
    df_weather_selected = df_weather_selected.withColumn('HourlyVisibility',
                           df_weather_selected["HourlyVisibility"].cast('double'))

    # WET BULB TEMPERATURE
    df_weather_selected = df_weather_selected.withColumn('HourlyWetBulbTemperature', 
                           split(df_weather_selected["HourlyWetBulbTemperature"],'s')[0])
    df_weather_selected = df_weather_selected.withColumn('HourlyWetBulbTemperature',
                           df_weather_selected["HourlyWetBulbTemperature"].cast('double'))

    # WIND DIRECTION
    df_weather_selected = df_weather_selected.withColumn('HourlyWindDirection', 
                           split(df_weather_selected["HourlyWindDirection"],'s')[0])
    # add new binary variable for calm winds, when hourly wind direction = 000
    df_weather_selected = df_weather_selected.withColumn('Calm_Winds',
                           when(df_weather_selected["HourlyWindDirection"] == '000', 1)\
                           .otherwise(0))
    df_weather_selected = df_weather_selected.withColumn('HourlyWindDirection',
                           when(df_weather_selected["HourlyWindDirection"] == 'VRB', None)\
                           .otherwise(df_weather_selected["HourlyWindDirection"])\
                           .cast('double'))

    # WIND GUST SPEED
    df_weather_selected = df_weather_selected.withColumn('HourlyWindGustSpeed', 
                           split(df_weather_selected["HourlyWindGustSpeed"],'s')[0])
    df_weather_selected = df_weather_selected.withColumn('HourlyWindGustSpeed',
                           df_weather_selected["HourlyWindGustSpeed"].cast('double'))


    # HOURLY WIND SPEED
    df_weather_selected = df_weather_selected.withColumn('HourlyWindSpeed', 
                           split(df_weather_selected["HourlyWindSpeed"],'s')[0])
    df_weather_selected = df_weather_selected.withColumn('HourlyWindSpeed',
                           df_weather_selected["HourlyWindSpeed"].cast('double'))

    # HOURLY SKY CONDITIONS
    df_weather_selected = df_weather_selected.withColumn('Sky_Conditions_CLR', 
                                                                   when(df_weather_selected['HourlySkyConditions'].contains('CLR'), 
                                                                        1).otherwise(0))
    df_weather_selected = df_weather_selected.withColumn('Sky_Conditions_FEW', 
                                                                   when(df_weather_selected['HourlySkyConditions'].contains('FEW'), 
                                                                        1).otherwise(0))
    df_weather_selected = df_weather_selected.withColumn('Sky_Conditions_SCT', 
                                                                   when(df_weather_selected['HourlySkyConditions'].contains('SCT'), 
                                                                        1).otherwise(0))
    df_weather_selected = df_weather_selected.withColumn('Sky_Conditions_BKN', 
                                                                   when(df_weather_selected['HourlySkyConditions'].contains('BKN'), 
                                                                        1).otherwise(0))
    df_weather_selected = df_weather_selected.withColumn('Sky_Conditions_OVC', 
                                                                   when(df_weather_selected['HourlySkyConditions'].contains('OVC'), 
                                                                        1).otherwise(0))
    df_weather_selected = df_weather_selected.withColumn('Sky_Conditions_VV', 
                                                                   when(df_weather_selected['HourlySkyConditions'].contains('VV'), 
                                                                        1).otherwise(0))

    # PRESENT WEATHER
    df_weather_selected = df_weather_selected.\
                               withColumn('Present_Weather_Drizzle', 
                                          when(df_weather_selected['HourlyPresentWeatherType'].contains('DZ'), 
                                               1).otherwise(0))
    df_weather_selected = df_weather_selected.\
                               withColumn('Present_Weather_Rain', 
                                          when(df_weather_selected['HourlyPresentWeatherType'].contains('RA'), 
                                               1).otherwise(0))
    df_weather_selected = df_weather_selected.\
                               withColumn('Present_Weather_Snow', 
                                          when(df_weather_selected['HourlyPresentWeatherType'].contains('SN'), 
                                               1).otherwise(0))
    df_weather_selected = df_weather_selected.\
                               withColumn('Present_Weather_SnowGrains', 
                                          when(df_weather_selected['HourlyPresentWeatherType'].contains('SG'), 
                                               1).otherwise(0))
    df_weather_selected = df_weather_selected.\
                               withColumn('Present_Weather_IceCrystals', 
                                          when(df_weather_selected['HourlyPresentWeatherType'].contains('IC'), 
                                               1).otherwise(0))
    df_weather_selected = df_weather_selected.\
                               withColumn('Present_Weather_Hail', 
                                          when((df_weather_selected['HourlyPresentWeatherType'].contains('PL')) | 
                                               (df_weather_selected['HourlyPresentWeatherType'].contains('GR')) |
                                               (df_weather_selected['HourlyPresentWeatherType'].contains('GS')) |
                                               (df_weather_selected['HourlyPresentWeatherType'].contains('HAIL')) |
                                               (df_weather_selected['HourlyPresentWeatherType'].contains('|27')) |
                                               (df_weather_selected['HourlyPresentWeatherType'].contains('SH:')), 
                                               1).otherwise(0))
    df_weather_selected = df_weather_selected.\
                               withColumn('Present_Weather_Mist', 
                                          when(df_weather_selected['HourlyPresentWeatherType'].contains('BR'), 
                                               1).otherwise(0))
    df_weather_selected = df_weather_selected.\
                               withColumn('Present_Weather_Fog', 
                                          when(df_weather_selected['HourlyPresentWeatherType'].contains('FG'), 
                                               1).otherwise(0))
    df_weather_selected = df_weather_selected.\
                               withColumn('Present_Weather_Smoke', 
                                          when(df_weather_selected['HourlyPresentWeatherType'].contains('FU'), 
                                               1).otherwise(0))
    df_weather_selected = df_weather_selected.\
                               withColumn('Present_Weather_Dust', 
                                          when((df_weather_selected['HourlyPresentWeatherType'].contains('FU')) |
                                               (df_weather_selected['HourlyPresentWeatherType'].contains('VA')) |
                                               (df_weather_selected['HourlyPresentWeatherType'].contains('DU')) |
                                               (df_weather_selected['HourlyPresentWeatherType'].contains('SA')) |
                                               (df_weather_selected['HourlyPresentWeatherType'].contains('PO')) |
                                               (df_weather_selected['HourlyPresentWeatherType'].contains('PY')) |
                                               (df_weather_selected['HourlyPresentWeatherType'].contains('SS')) |
                                               (df_weather_selected['HourlyPresentWeatherType'].contains('DS')), 
                                               1).otherwise(0))
    df_weather_selected = df_weather_selected.\
                               withColumn('Present_Weather_Haze', 
                                          when(df_weather_selected['HourlyPresentWeatherType'].contains('HZ'), 
                                               1).otherwise(0))
    df_weather_selected = df_weather_selected.\
                               withColumn('Present_Weather_Storm', 
                                          when((df_weather_selected['HourlyPresentWeatherType'].contains('SQ')) |
                                               (df_weather_selected['HourlyPresentWeatherType'].contains('FC')) | 
                                               (df_weather_selected['HourlyPresentWeatherType'].contains('TS')),
                                               1).otherwise(0))
    

    # DROP COLUMNS THAT WE'RE NO LONGER USING
    df_weather_selected = df_weather_selected.drop(*drop_cols)
    df_weather_selected.createOrReplaceTempView("vw_weather_selected")
    display(df_weather_selected)
    df_weather_selected.columns

load_and_transform_data()

# COMMAND ----------

# MAGIC %md ## Merge Flights With Weather

# COMMAND ----------

def save_hourly_weather():
    """
        Save the hourly weather data. Each field of interest is appropriately aggregated over one hour window.
    """
    ########## MODIFIED QUERY TO ADDRESS DATA LEAKAGE ISSUES ##########
    sql = """
        SELECT STATION, DATEHOUR, AVG(ELEVATION) as Avg_Elevation,
        AVG(HourlyAltimeterSetting) as Avg_HourlyAltimeterSetting,
        AVG(HourlyDewPointTemperature) as Avg_HourlyDewPointTemperature,
        AVG(HourlyDryBulbTemperature) as Avg_HourlyDryBulbTemperature, 
        AVG(HourlyPressureChange) as Avg_HourlyPressureChange, 
        AVG(HourlyRelativeHumidity) as Avg_HourlyRelativeHumidity, 
        AVG(HourlySeaLevelPressure) as Avg_HourlySeaLevelPressure,
        AVG(HourlyStationPressure) as Avg_HourlyStationPressure, 
        AVG(HourlyVisibility) as Avg_HourlyVisibility, 
        AVG(HourlyWetBulbTemperature) as Avg_HourlyWetBulbTemperature, 
        AVG(HourlyWindDirection) as Avg_HourlyWindDirection,
        AVG(HourlyWindGustSpeed) as Avg_HourlyWindGustSpeed, 
        AVG(HourlyWindSpeed) as Avg_HourlyWindSpeed, 
        AVG(Precip_Double) as Avg_Precip_Double, 
        MAX(Trace_Rain) as Trace_Rain, 
        MAX(NonZero_Rain) as NonZero_Rain, 
        MAX(HourlyPressureTendency_Increasing) as HourlyPressureTendency_Increasing, 
        MAX(HourlyPressureTendency_Decreasing) as HourlyPressureTendency_Decreasing, 
        MAX(HourlyPressureTendency_Constant) as HourlyPressureTendency_Constant,
        MAX(Calm_Winds) as Calm_Winds, 
        MAX(Sky_Conditions_CLR)  as Sky_Conditions_CLR, 
        MAX(Sky_Conditions_FEW) as Sky_Conditions_FEW, 
        MAX(Sky_Conditions_SCT) as Sky_Conditions_SCT, 
        MAX(Sky_Conditions_BKN) as Sky_Conditions_BKN, 
        MAX(Sky_Conditions_OVC) as Sky_Conditions_OVC, 
        MAX(Sky_Conditions_VV) as Sky_Conditions_VV, 
        MAX(Present_Weather_Drizzle) as Present_Weather_Drizzle, 
        MAX(Present_Weather_Rain) as Present_Weather_Rain, 
        MAX(Present_Weather_Snow) as Present_Weather_Snow, 
        MAX(Present_Weather_SnowGrains) as Present_Weather_SnowGrains, 
        MAX(Present_Weather_IceCrystals) as Present_Weather_IceCrystals, 
        MAX(Present_Weather_Hail) as Present_Weather_Hail, 
        MAX(Present_Weather_Mist) as Present_Weather_Mist, 
        MAX(Present_Weather_Fog) as Present_Weather_Fog, 
        MAX(Present_Weather_Smoke) as Present_Weather_Smoke, 
        MAX(Present_Weather_Dust) as Present_Weather_Dust,
        MAX(Present_Weather_Haze) as Present_Weather_Haze, 
        MAX(Present_Weather_Storm) as Present_Weather_Storm 
        FROM vw_weather_selected
        GROUP BY STATION, DATEHOUR
        ORDER BY STATION, DATEHOUR
    """
    df = spark.sql(sql)
    df.write.mode('overwrite').parquet(CLEAN_WEATHER_LOC)
    display(df)  
    
save_hourly_weather()

# COMMAND ----------

def load_views_for_staging_merged_data():
    flight_fields_to_ignore = ['DIV_AIRPORT_LANDINGS', 'DIV_REACHED_DEST', 'DIV_ACTUAL_ELAPSED_TIME', 'DIV_ARR_DELAY', 'DIV_DISTANCE', 'DIV1_AIRPORT', 'DIV1_AIRPORT_ID', 'DIV1_AIRPORT_SEQ_ID', 'DIV1_WHEELS_ON', 'DIV1_TOTAL_GTIME', 'DIV1_LONGEST_GTIME', 'DIV1_WHEELS_OFF', 'DIV1_TAIL_NUM', 'DIV2_AIRPORT', 'DIV2_AIRPORT_ID', 'DIV2_AIRPORT_SEQ_ID', 'DIV2_WHEELS_ON', 'DIV2_TOTAL_GTIME', 'DIV2_LONGEST_GTIME', 'DIV2_WHEELS_OFF', 'DIV2_TAIL_NUM', 'DIV3_AIRPORT', 'DIV3_AIRPORT_ID', 'DIV3_AIRPORT_SEQ_ID', 'DIV3_WHEELS_ON', 'DIV3_TOTAL_GTIME', 'DIV3_LONGEST_GTIME', 'DIV3_WHEELS_OFF', 'DIV3_TAIL_NUM', 'DIV4_AIRPORT', 'DIV4_AIRPORT_ID', 'DIV4_AIRPORT_SEQ_ID', 'DIV4_WHEELS_ON', 'DIV4_TOTAL_GTIME', 'DIV4_LONGEST_GTIME', 'DIV4_WHEELS_OFF', 'DIV4_TAIL_NUM', 'DIV5_AIRPORT', 'DIV5_AIRPORT_ID', 'DIV5_AIRPORT_SEQ_ID', 'DIV5_WHEELS_ON', 'DIV5_TOTAL_GTIME', 'DIV5_LONGEST_GTIME', 'DIV5_WHEELS_OFF', 'DIV5_TAIL_NUM', '_c109', '_local_year']
    
    spark.read.parquet(AIRPORTS_MASTER_LOC).createOrReplaceTempView("vw_airports_master")
    spark.read.parquet(AIRPORTS_WS_LOC).createOrReplaceTempView("vw_airports_ws")
    spark.read.parquet(CLEAN_WEATHER_LOC).createOrReplaceTempView("vw_weather_cleaned")

    df_pre_covid_flights = spark.read.parquet(FLIGHT_PRE_COVID_RAW_LOC)
    
    df_covid_flights = spark.read.parquet(FLIGHT_COVID_RAW_LOC)
    for index in range(len(df_pre_covid_flights.columns)):
        df_covid_flights = df_covid_flights.withColumnRenamed(df_covid_flights.columns[index], df_pre_covid_flights.columns[index])
    df_covid_flights = df_covid_flights.drop(*flight_fields_to_ignore)
    df_covid_flights.createOrReplaceTempView("vw_covid_flights")
    df_pre_covid_flights = df_pre_covid_flights.drop(*flight_fields_to_ignore)
    df_pre_covid_flights.createOrReplaceTempView("vw_pre_covid_flights")
    
    spark.sql("CACHE LAZY TABLE vw_airports_ws")
    spark.sql("CACHE LAZY TABLE vw_airports_master")
    spark.sql("UNCACHE TABLE IF EXISTS vw_weather_cleaned")
    spark.sql("UNCACHE TABLE IF EXISTS vw_pre_covid_flights")
    
load_views_for_staging_merged_data()



# COMMAND ----------

def to_map(schema):
    columns = {}
    for col in schema:
        columns[col.name] = col.dataType
    return columns

def to_set(schema):
    columns = set()
    for column in schema:
        columns.add(f'{column.name}::{column.dataType}')
    return columns

def check_columns():
    df_pre_covid_flights = spark.sql("SELECT * from vw_pre_covid_flights")
#     pre_covid_cols = to_map(df_pre_covid_flights.schema)
    pre_covid_set = to_set(df_pre_covid_flights.schema)

    df_covid_flights = spark.sql("SELECT * from vw_covid_flights")
    covid_set = to_set(df_covid_flights.schema)
#     covid_cols = to_map(df_covid_flights.schema)    
#     print(pre_covid_cols)
#     print(covid_cols)
    print('---------------')
    print(pre_covid_set ^ covid_set)

    print('---------------')
    print(pre_covid_set)
    print('---------------')
    print(covid_set)
    print('---------------')
    
check_columns()

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from vw_weather_cleaned limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC         WITH flights as (
# MAGIC             SELECT DISTINCT concat(right(concat('0000', CRS_DEP_TIME), 4), '00') as _dep_time_str,
# MAGIC               to_timestamp(concat(FL_DATE, ' ', right(concat('0000', CRS_DEP_TIME), 4), '00'), 'yyyy-MM-dd Hmmss') as _local_dept_ts, 
# MAGIC               to_timestamp(concat(FL_DATE, ' ', right(concat('0000', CRS_DEP_TIME), 4), '00'), 'yyyy-MM-dd Hmmss') - INTERVAL 2 HOUR as _local_dept_minus2_ts,
# MAGIC               to_timestamp(concat(FL_DATE, ' ', right(concat('0000', CRS_DEP_TIME), 4), '00'), 'yyyy-MM-dd Hmmss') + make_interval(0, 0, 0, 0, 0, DEP_DELAY) as _local_dept_actual_ts, 
# MAGIC               to_timestamp(concat(FL_DATE, ' ', right(concat('0000', CRS_DEP_TIME), 4), '00'), 'yyyy-MM-dd Hmmss') + make_interval(0, 0, 0, 0, 0, CRS_ELAPSED_TIME) as _local_at_src_airport_arr_ts,
# MAGIC               to_timestamp(concat(FL_DATE, ' ', right(concat('0000', CRS_DEP_TIME), 4), '00'), 'yyyy-MM-dd Hmmss') + make_interval(0, 0, 0, 0, 0, DEP_DELAY + ACTUAL_ELAPSED_TIME) as _local_at_src_airport_arr_actual_ts,
# MAGIC               *
# MAGIC               FROM vw_pre_covid_flights 
# MAGIC               WHERE YEAR=2015)
# MAGIC           , flights_airport as (
# MAGIC               SELECT 
# MAGIC                   to_utc_timestamp(date_format(_local_dept_ts , "yyyy-MM-dd HH:mm:ss"), origin_airport.tz) as _utc_dept_ts,
# MAGIC                   to_utc_timestamp(date_format(_local_dept_minus2_ts , "yyyy-MM-dd HH:mm:ss"), origin_airport.tz) as _utc_dept_minus2_ts,
# MAGIC                   to_utc_timestamp(date_format(_local_dept_actual_ts , "yyyy-MM-dd HH:mm:ss"), origin_airport.tz) as _utc_dept_actual_ts,
# MAGIC                   to_utc_timestamp(date_format(_local_at_src_airport_arr_ts , "yyyy-MM-dd HH:mm:ss"), origin_airport.tz) as _utc_arr_ts,
# MAGIC                   to_utc_timestamp(date_format(_local_at_src_airport_arr_actual_ts , "yyyy-MM-dd HH:mm:ss"), origin_airport.tz) as _utc_arr_actual_ts,
# MAGIC                   flights.*, 
# MAGIC                   origin_airport.iata as origin_airport_iata, origin_airport.tz as origin_airport_tz, 
# MAGIC                   origin_airport.type as origin_airport_type, origin_airport.elevation as origin_airport_elevation,
# MAGIC                   origin_airport.iso_country as origin_airport_iso_country, origin_airport.iso_region as origin_airport_iso_region, 
# MAGIC                   origin_airport.ws_station_id as origin_airport_ws_station_id,
# MAGIC                   dest_airport.iata as dest_airport_iata, dest_airport.tz as dest_airport_tz, 
# MAGIC                   dest_airport.type as dest_airport_type, dest_airport.elevation as dest_airport_elevation,
# MAGIC                   dest_airport.iso_country as dest_airport_iso_country, dest_airport.iso_region as dest_airport_iso_region, 
# MAGIC                   dest_airport.ws_station_id as dest_airport_ws_station_id
# MAGIC               FROM flights
# MAGIC                   JOIN vw_airports_master as origin_airport ON flights.ORIGIN = origin_airport.iata
# MAGIC                   JOIN vw_airports_master as dest_airport ON flights.DEST = dest_airport.iata)
# MAGIC SELECT CRS_DEP_TIME,_dep_time_str,DEP_TIME,DEP_DELAY,_utc_dept_ts,_utc_dept_minus2_ts,_utc_dept_actual_ts,_local_dept_ts,_local_dept_minus2_ts,_local_dept_actual_ts,CRS_ARR_TIME,ARR_TIME,ARR_DELAY,CRS_ELAPSED_TIME,ACTUAL_ELAPSED_TIME,_utc_arr_ts,_utc_arr_actual_ts,_local_at_src_airport_arr_ts,_local_at_src_airport_arr_actual_ts,ORIGIN,DEST,origin_airport_iata,origin_airport_tz,dest_airport_iata,dest_airport_tz
# MAGIC FROM flights_airport 
# MAGIC LIMIT 10

# COMMAND ----------

def create_joined_data(flight_view, year_start, year_end, destination):
    """
        Joins flight data to airport and weather data. flight_view should be the view with the flight source.
        year_start and year_end (both inclusive) is the period for which we want data staged.
        destination is the location where the data needs to be written to.
        
        Note: The data written out will be partitioned by year and any existing data in destination will be overwritten.
        Also, the year_start and year_end are treated as year based on local timezone.
    """
    sql = """
        WITH flights as (
            SELECT DISTINCT concat(right(concat('0000', CRS_DEP_TIME), 4), '00') as _dep_time_str,
              to_timestamp(concat(FL_DATE, ' ', right(concat('0000', CRS_DEP_TIME), 4), '00'), 'yyyy-MM-dd Hmmss') as _local_dept_ts, 
              to_timestamp(concat(FL_DATE, ' ', right(concat('0000', CRS_DEP_TIME), 4), '00'), 'yyyy-MM-dd Hmmss') - INTERVAL 2 HOUR as _local_dept_minus2_ts,
              to_timestamp(concat(FL_DATE, ' ', right(concat('0000', CRS_DEP_TIME), 4), '00'), 'yyyy-MM-dd Hmmss') + make_interval(0, 0, 0, 0, 0, DEP_DELAY) as _local_dept_actual_ts, 
              to_timestamp(concat(FL_DATE, ' ', right(concat('0000', CRS_DEP_TIME), 4), '00'), 'yyyy-MM-dd Hmmss') + make_interval(0, 0, 0, 0, 0, CRS_ELAPSED_TIME) as _local_at_src_airport_arr_ts,
              to_timestamp(concat(FL_DATE, ' ', right(concat('0000', CRS_DEP_TIME), 4), '00'), 'yyyy-MM-dd Hmmss') + make_interval(0, 0, 0, 0, 0, DEP_DELAY + ACTUAL_ELAPSED_TIME) as _local_at_src_airport_arr_actual_ts,
              *
              FROM """ + flight_view + """
              WHERE cast(YEAR as int) >= """ + str(year_start) + """ AND cast(YEAR as int) <= """ + str(year_end) + """)
          , flights_airport as (
              SELECT 
                  to_utc_timestamp(date_format(_local_dept_ts , "yyyy-MM-dd HH:mm:ss"), origin_airport.tz) as _utc_dept_ts,
                  to_utc_timestamp(date_format(_local_dept_minus2_ts , "yyyy-MM-dd HH:mm:ss"), origin_airport.tz) as _utc_dept_minus2_ts,
                  to_utc_timestamp(date_format(_local_dept_actual_ts , "yyyy-MM-dd HH:mm:ss"), origin_airport.tz) as _utc_dept_actual_ts,
                  to_utc_timestamp(date_format(_local_at_src_airport_arr_ts , "yyyy-MM-dd HH:mm:ss"), origin_airport.tz) as _utc_arr_ts,
                  to_utc_timestamp(date_format(_local_at_src_airport_arr_actual_ts , "yyyy-MM-dd HH:mm:ss"), origin_airport.tz) as _utc_arr_actual_ts,
                  flights.*, 
                  origin_airport.iata as origin_airport_iata, origin_airport.tz as origin_airport_tz, 
                  origin_airport.type as origin_airport_type, origin_airport.elevation as origin_airport_elevation,
                  origin_airport.iso_country as origin_airport_iso_country, origin_airport.iso_region as origin_airport_iso_region, 
                  origin_airport.ws_station_id as origin_airport_ws_station_id,
                  dest_airport.iata as dest_airport_iata, dest_airport.tz as dest_airport_tz, 
                  dest_airport.type as dest_airport_type, dest_airport.elevation as dest_airport_elevation,
                  dest_airport.iso_country as dest_airport_iso_country, dest_airport.iso_region as dest_airport_iso_region, 
                  dest_airport.ws_station_id as dest_airport_ws_station_id
              FROM flights
                  JOIN vw_airports_master as origin_airport ON flights.ORIGIN = origin_airport.iata
                  JOIN vw_airports_master as dest_airport ON flights.DEST = dest_airport.iata)
          , merged as (
            SELECT 
                flights_airport.*,
                    origin_weather.STATION as origin_weather_Station,
                    origin_weather.DATEHOUR as origin_weather_Datehour,
                    origin_weather.Avg_Elevation as origin_weather_Avg_Elevation,
                    origin_weather.Avg_HourlyAltimeterSetting as origin_weather_Avg_HourlyAltimeterSetting,
                    origin_weather.Avg_HourlyDewPointTemperature as origin_weather_Avg_HourlyDewPointTemperature,
                    origin_weather.Avg_HourlyDryBulbTemperature as origin_weather_Avg_HourlyDryBulbTemperature,
                    origin_weather.Avg_HourlyPressureChange as origin_weather_Avg_HourlyPressureChange,
                    origin_weather.Avg_HourlyRelativeHumidity as origin_weather_Avg_HourlyRelativeHumidity,
                    origin_weather.Avg_HourlySeaLevelPressure as origin_weather_Avg_HourlySeaLevelPressure,
                    origin_weather.Avg_HourlyStationPressure as origin_weather_Avg_HourlyStationPressure,
                    origin_weather.Avg_HourlyVisibility as origin_weather_Avg_HourlyVisibility,
                    origin_weather.Avg_HourlyWetBulbTemperature as origin_weather_Avg_HourlyWetBulbTemperature,
                    origin_weather.Avg_HourlyWindDirection as origin_weather_Avg_HourlyWindDirection,
                    origin_weather.Avg_HourlyWindGustSpeed as origin_weather_Avg_HourlyWindGustSpeed,
                    origin_weather.Avg_HourlyWindSpeed as origin_weather_Avg_HourlyWindSpeed,
                    origin_weather.Avg_Precip_Double as origin_weather_Avg_Precip_Double,
                    origin_weather.Trace_Rain as origin_weather_Trace_Rain,
                    origin_weather.NonZero_Rain as origin_weather_NonZero_Rain,
                    origin_weather.HourlyPressureTendency_Increasing as origin_weather_HourlyPressureTendency_Increasing,
                    origin_weather.HourlyPressureTendency_Decreasing as origin_weather_HourlyPressureTendency_Decreasing,
                    origin_weather.HourlyPressureTendency_Constant as origin_weather_HourlyPressureTendency_Constant,
                    origin_weather.Calm_Winds as origin_weather_Calm_Winds,
                    origin_weather.Sky_Conditions_CLR as origin_weather_Sky_Conditions_CLR,
                    origin_weather.Sky_Conditions_FEW as origin_weather_Sky_Conditions_FEW,
                    origin_weather.Sky_Conditions_SCT as origin_weather_Sky_Conditions_SCT,
                    origin_weather.Sky_Conditions_BKN as origin_weather_Sky_Conditions_BKN,
                    origin_weather.Sky_Conditions_OVC as origin_weather_Sky_Conditions_OVC,
                    origin_weather.Sky_Conditions_VV as origin_weather_Sky_Conditions_VV,
                    origin_weather.Present_Weather_Drizzle as origin_weather_Present_Weather_Drizzle,
                    origin_weather.Present_Weather_Rain as origin_weather_Present_Weather_Rain,
                    origin_weather.Present_Weather_Snow as origin_weather_Present_Weather_Snow,
                    origin_weather.Present_Weather_SnowGrains as origin_weather_Present_Weather_SnowGrains,
                    origin_weather.Present_Weather_IceCrystals as origin_weather_Present_Weather_IceCrystals,
                    origin_weather.Present_Weather_Hail as origin_weather_Present_Weather_Hail,
                    origin_weather.Present_Weather_Mist as origin_weather_Present_Weather_Mist,
                    origin_weather.Present_Weather_Fog as origin_weather_Present_Weather_Fog,
                    origin_weather.Present_Weather_Smoke as origin_weather_Present_Weather_Smoke,
                    origin_weather.Present_Weather_Dust as origin_weather_Present_Weather_Dust,
                    origin_weather.Present_Weather_Haze as origin_weather_Present_Weather_Haze,
                    origin_weather.Present_Weather_Storm as origin_weather_Present_Weather_Storm,
                    dest_weather.STATION as dest_weather_Station,
                    dest_weather.DATEHOUR as dest_weather_Datehour,
                    dest_weather.Avg_Elevation as dest_weather_Avg_Elevation,
                    dest_weather.Avg_HourlyAltimeterSetting as dest_weather_Avg_HourlyAltimeterSetting,
                    dest_weather.Avg_HourlyDewPointTemperature as dest_weather_Avg_HourlyDewPointTemperature,
                    dest_weather.Avg_HourlyDryBulbTemperature as dest_weather_Avg_HourlyDryBulbTemperature,
                    dest_weather.Avg_HourlyPressureChange as dest_weather_Avg_HourlyPressureChange,
                    dest_weather.Avg_HourlyRelativeHumidity as dest_weather_Avg_HourlyRelativeHumidity,
                    dest_weather.Avg_HourlySeaLevelPressure as dest_weather_Avg_HourlySeaLevelPressure,
                    dest_weather.Avg_HourlyStationPressure as dest_weather_Avg_HourlyStationPressure,
                    dest_weather.Avg_HourlyVisibility as dest_weather_Avg_HourlyVisibility,
                    dest_weather.Avg_HourlyWetBulbTemperature as dest_weather_Avg_HourlyWetBulbTemperature,
                    dest_weather.Avg_HourlyWindDirection as dest_weather_Avg_HourlyWindDirection,
                    dest_weather.Avg_HourlyWindGustSpeed as dest_weather_Avg_HourlyWindGustSpeed,
                    dest_weather.Avg_HourlyWindSpeed as dest_weather_Avg_HourlyWindSpeed,
                    dest_weather.Avg_Precip_Double as dest_weather_Avg_Precip_Double,
                    dest_weather.Trace_Rain as dest_weather_Trace_Rain,
                    dest_weather.NonZero_Rain as dest_weather_NonZero_Rain,
                    dest_weather.HourlyPressureTendency_Increasing as dest_weather_HourlyPressureTendency_Increasing,
                    dest_weather.HourlyPressureTendency_Decreasing as dest_weather_HourlyPressureTendency_Decreasing,
                    dest_weather.HourlyPressureTendency_Constant as dest_weather_HourlyPressureTendency_Constant,
                    dest_weather.Calm_Winds as dest_weather_Calm_Winds,
                    dest_weather.Sky_Conditions_CLR as dest_weather_Sky_Conditions_CLR,
                    dest_weather.Sky_Conditions_FEW as dest_weather_Sky_Conditions_FEW,
                    dest_weather.Sky_Conditions_SCT as dest_weather_Sky_Conditions_SCT,
                    dest_weather.Sky_Conditions_BKN as dest_weather_Sky_Conditions_BKN,
                    dest_weather.Sky_Conditions_OVC as dest_weather_Sky_Conditions_OVC,
                    dest_weather.Sky_Conditions_VV as dest_weather_Sky_Conditions_VV,
                    dest_weather.Present_Weather_Drizzle as dest_weather_Present_Weather_Drizzle,
                    dest_weather.Present_Weather_Rain as dest_weather_Present_Weather_Rain,
                    dest_weather.Present_Weather_Snow as dest_weather_Present_Weather_Snow,
                    dest_weather.Present_Weather_SnowGrains as dest_weather_Present_Weather_SnowGrains,
                    dest_weather.Present_Weather_IceCrystals as dest_weather_Present_Weather_IceCrystals,
                    dest_weather.Present_Weather_Hail as dest_weather_Present_Weather_Hail,
                    dest_weather.Present_Weather_Mist as dest_weather_Present_Weather_Mist,
                    dest_weather.Present_Weather_Fog as dest_weather_Present_Weather_Fog,
                    dest_weather.Present_Weather_Smoke as dest_weather_Present_Weather_Smoke,
                    dest_weather.Present_Weather_Dust as dest_weather_Present_Weather_Dust,
                    dest_weather.Present_Weather_Haze as dest_weather_Present_Weather_Haze,
                    dest_weather.Present_Weather_Storm as dest_weather_Present_Weather_Storm
            FROM flights_airport flights_airport
              JOIN vw_weather_cleaned origin_weather 
                ON origin_weather.STATION = flights_airport.origin_airport_ws_station_id 
                  AND date(_utc_dept_minus2_ts) = date(origin_weather.DATEHOUR) AND hour(_utc_dept_minus2_ts) = hour(origin_weather.DATEHOUR)
              JOIN vw_weather_cleaned dest_weather 
                ON dest_weather.STATION = flights_airport.dest_airport_ws_station_id 
                  AND date(_utc_dept_minus2_ts) = date(dest_weather.DATEHOUR) AND hour(_utc_dept_minus2_ts) = hour(dest_weather.DATEHOUR)
          )

        SELECT * FROM merged
    """
    df = spark.sql(sql)
    display(df)
    df.repartition(col('YEAR')).write.partitionBy('YEAR').mode('overwrite').parquet(destination)
    
create_joined_data('vw_pre_covid_flights', 2015, 2018, configuration.FINAL_JOINED_DATA_2015_2018)
create_joined_data('vw_pre_covid_flights', 2019, 2019, configuration.FINAL_JOINED_DATA_2019)
create_joined_data('vw_covid_flights', 2020, 2020, configuration.FINAL_JOINED_DATA_2020)
create_joined_data('vw_covid_flights', 2021, 2021, configuration.FINAL_JOINED_DATA_2021)

# COMMAND ----------

# TODO Need to look at 900+ rows with CANCELLED <> 1 and DEP_DEL15 null.
# display(spark.read.parquet(FINAL_JOINED_DATA_2015_2018).filter(col('CANCELLED') != 0).filter(col('DEP_DEL15').isNull()))


# COMMAND ----------

def pull_joined():
    df = spark.read.parquet(configuration.FINAL_JOINED_DATA_2015_2018)
    display(df)
    print(f'Total number of flights: {df.count()}')
    display(df.filter((col('FL_DATE') == '2015-01-01') & (col('ORIGIN') == 'SFO')))
    
pull_joined()


# COMMAND ----------

def register_joined_data_view():
    df = spark.read.parquet(configuration.FINAL_JOINED_DATA_2015_2018)
    df.createOrReplaceTempView("vw_joined_2015_2018")
    
    df = spark.read.parquet(configuration.FINAL_JOINED_DATA_2019)
    df.createOrReplaceTempView("vw_joined_2019")

register_joined_data_view()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT year(DATEHOUR), count(*)
# MAGIC FROM vw_weather_cleaned 
# MAGIC GROUP BY year(DATEHOUR)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT TAIL_NUM, _utc_dept_ts, count(*) as counts
# MAGIC FROM vw_joined_2015_2018 
# MAGIC WHERE TAIL_NUM is not null and cancelled <> 1 and DEP_DEL15 = 0
# MAGIC GROUP BY TAIL_NUM, _utc_dept_ts
# MAGIC HAVING counts > 1
# MAGIC ORDER by counts desc

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH dupe_flights as (
# MAGIC   SELECT TAIL_NUM, _utc_dept_ts, count(*) as counts
# MAGIC   FROM vw_joined_2015_2018 
# MAGIC   WHERE TAIL_NUM is not null and cancelled <> 1 and DEP_DEL15 = 1
# MAGIC   GROUP BY TAIL_NUM, _utc_dept_ts
# MAGIC   HAVING counts > 1
# MAGIC   ORDER by counts desc
# MAGIC )
# MAGIC SELECT counts as dup_counts, a._utc_dept_ts, FL_DATE, a.TAIL_NUM, OP_CARRIER_FL_NUM, OP_UNIQUE_CARRIER, OP_CARRIER_AIRLINE_ID, 
# MAGIC   OP_CARRIER, ORIGIN, DEST, CRS_DEP_TIME, DEP_TIME, DEP_DELAY, CRS_ARR_TIME, ARR_TIME, ARR_DELAY
# MAGIC FROM vw_joined_2015_2018 a join dupe_flights b on a.TAIL_NUM = b.TAIL_NUM and a._utc_dept_ts = b._utc_dept_ts
# MAGIC ORDER by TAIL_NUM, _utc_dept_ts

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT FL_DATE, TAIL_NUM, OP_UNIQUE_CARRIER, OP_CARRIER_FL_NUM, ORIGIN, ORIGIN_CITY_NAME, DEST, DEST_CITY_NAME, DEP_DEL15
# MAGIC FROM vw_joined_2015_2018
# MAGIC WHERE TAIL_NUM in ('ALL', 'PLANET', 'D942DN', 'N0EGMQ', 'SS25')

# COMMAND ----------

# MAGIC %md #Stand Up Dev Data

# COMMAND ----------

def create_dev_data():
    sql = """
        SELECT * FROM vw_joined_2015_2018 
        WHERE YEAR = 2015 AND FL_DATE >= '2015-01-01' AND FL_DATE < '2015-04-01'
    """
    df = spark.sql(sql)
    df.coalesce(1).write.mode('overwrite').parquet(FINAL_JOINED_DATA_2015_Q1)


create_dev_data()

# COMMAND ----------

# Create toy dataset with Q1 data from 2015-2018 and 2019.
def create_toy_dataset():
    sql = """
        SELECT * FROM vw_joined_2015_2018 
        WHERE YEAR = 2015 AND FL_DATE >= '2015-01-01' AND FL_DATE < '2015-04-01'
        UNION
        SELECT * FROM vw_joined_2015_2018 
        WHERE YEAR = 2016 AND FL_DATE >= '2016-01-01' AND FL_DATE < '2016-04-01'
        UNION
        SELECT * FROM vw_joined_2015_2018 
        WHERE YEAR = 2017 AND FL_DATE >= '2017-01-01' AND FL_DATE < '2017-04-01'        
        UNION
        SELECT * FROM vw_joined_2015_2018 
        WHERE YEAR = 2018 AND FL_DATE >= '2018-01-01' AND FL_DATE < '2018-04-01'        
    """
    df = spark.sql(sql)
    df.coalesce(1).write.mode('overwrite').parquet(configuration.FINAL_JOINED_DATA_Q1_2015_2018)
    
    sql = """
        SELECT * FROM vw_joined_2019 
        WHERE YEAR = 2019 AND FL_DATE >= '2019-01-01' AND FL_DATE < '2019-04-01'
    """
    df = spark.sql(sql)
    df.coalesce(1).write.mode('overwrite').parquet(FINAL_JOINED_DATA_Q1_2019)

create_toy_dataset()
