# Databricks notebook source
# MAGIC %md
# MAGIC # Weather Cleaning with New Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pyspark
import datetime as dt
from pyspark.sql.functions import when, to_timestamp, year, date_trunc, split, regexp_replace, array_max, length, substring, greatest, minute, hour, expr

blob_container = "w261team11" # The name of your container created in https://portal.azure.com
storage_account = "w261sa" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261team11" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261team11key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
WEATHER_LOC = f"{blob_url}/staged/weather"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather Data

# COMMAND ----------

def load_and_transform_data():
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
    closest_stations = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/jconde@berkeley.edu/closest_ws_stations.csv")
    display(closest_stations)
    closest_stations.createOrReplaceTempView("closest_stations")    
    
    # Convert DATE Column from string to timestamp
    # df_weather_2015_selected = df_weather_2015_selected.withColumn("DATE", to_timestamp(col("DATE"), "yyyy-MM-dd'T'HH:mm:ss"))
    df_weather_selected = df_weather_selected.withColumn("DATE", df_weather_selected['DATE'].cast('timestamp'))
    # Create new column with the hour of the weather reading (get rid of the minute reading)
    df_weather_selected = df_weather_selected.withColumn('DATEHOUR', date_trunc("hour", df_weather_selected["DATE"]))

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


def save_hourly_weather():
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
        RIGHT JOIN closest_stations ON vw_weather_selected.STATION = closest_stations.ws_id
        GROUP BY STATION, DATEHOUR
        ORDER BY STATION, DATEHOUR
    """
    df = spark.sql(sql)
    df.write.mode('overwrite').parquet(WEATHER_LOC + '/clean_weather_data.parquet')
    display(df)  
    
save_hourly_weather()

# COMMAND ----------

df_weather.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Prevent Data Leakage

# COMMAND ----------

AIRPORTS_WS_LOC = f"{blob_url}/staged/airports_weatherstations"
df_airports_ws = spark.read.parquet(f"{AIRPORTS_WS_LOC}")
display(df_airports_ws)

# COMMAND ----------

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
closest_stations = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/jconde@berkeley.edu/closest_ws_stations.csv")
display(closest_stations)
closest_stations.createOrReplaceTempView("closest_stations")    

# Convert DATE Column from string to timestamp
# df_weather_2015_selected = df_weather_2015_selected.withColumn("DATE", to_timestamp(col("DATE"), "yyyy-MM-dd'T'HH:mm:ss"))
df_weather_selected = df_weather_selected.withColumn("DATE", df_weather_selected['DATE'].cast('timestamp'))
# Create new column with the hour of the weather reading (get rid of the minute reading)
# df_weather_selected = df_weather_selected.withColumn('DATEHOUR', date_trunc("hour", df_weather_selected["DATE"]))

# COMMAND ----------

def run_sql():
    sql = """
        SELECT COUNT(DISTINCT(STATION))
        FROM vw_weather_selected
    """
    
    result = spark.sql(sql)
    display(result)
    
run_sql()

# COMMAND ----------

def run_sql():
    sql = """
        SELECT COUNT(DISTINCT(ws_id))
        FROM closest_stations
    """
    
    result = spark.sql(sql)
    display(result)
    
run_sql()

# COMMAND ----------

test_cols = ['DATE']
test = df_weather.select(*test_cols)

# COMMAND ----------

test = test.withColumn("MINS", when(minute(test['DATE']) > 0, 
                                    minute(test['DATE'])-60).otherwise(0))
test = test.withColumn("CORRECTED HR", when(test["MINS"] < 0, hour(test["DATE"])+1).otherwise(hour(test["DATE"])))
display(test)

# COMMAND ----------

df_weather_selected = df_weather_selected.withColumn("MINS", when(minute(df_weather_selected['DATE']) > 0, 
                                    minute(df_weather_selected['DATE'])-60).otherwise(0))
df_weather_selected = df_weather_selected.withColumn("CORRECTED_DATE", when(df_weather_selected["MINS"] < 0, 
                                                                          df_weather_selected["DATE"] + expr("INTERVAL 1 HOUR"))\
                                                     .otherwise(df_weather_selected["DATE"]))
display(df_weather_selected)
df_weather_selected.createOrReplaceTempView("weather_test")   

# COMMAND ----------

def stage_weather_data():
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
    
    ######## NEW COLUMNS TO ADD FOR TIME CORRECTION ##############
    
    df_weather_selected = df_weather_selected.withColumn("MINS", when(minute(df_weather_selected['DATE']) > 0, 
                                    minute(df_weather_selected['DATE'])-60).otherwise(0))
    df_weather_selected = df_weather_selected.withColumn("DATE_PLUS_HOUR", when(df_weather_selected["MINS"] < 0, 
                                                                          df_weather_selected["DATE"] + expr("INTERVAL 1 HOUR"))\
                                                     .otherwise(df_weather_selected["DATE"]))
    
    #############################################################
    
    
    ############### MODIFIED QUERY FOR stage_weather_data ##############################
    sql = """
            SELECT date(DATE_PLUS_HOUR) as _utc_date, hour(DATE_PLUS_HOUR) as _utc_hour_of_day, * 
            FROM 
              (SELECT *, row_number() OVER (PARTITION BY STATION, date(DATE_PLUS_HOUR), hour(DATE_PLUS_HOUR) ORDER BY MINS DESC) as rn FROM weather_test) tmp
            WHERE rn = 1 
            ORDER BY STATION, _utc_date, _utc_hour_of_day
    """
    
    #############################################################

       
#     closest_stations = spark.read.format("csv").option("header",                                               "true").load("dbfs:/FileStore/shared_uploads/jconde@berkeley.edu/closest_ws_stations.csv")
#     closest_stations.createOrReplaceTempView("closest_stations")
    
    
    df = spark.sql(sql)
    df.createOrReplaceTempView("corrected_hours")
    
    AIRPORTS_WS_LOC = f"{blob_url}/staged/airports_weatherstations"
    df_airports_ws = spark.read.parquet(f"{AIRPORTS_WS_LOC}")
#     display(df_airports_ws)
    df_airports_ws.createOrReplaceTempView("airports_ws")
    
    sql2 = """
        SELECT *
        FROM corrected_hours
        JOIN airports_ws ON corrected_hours.STATION = airports_ws.ws_station_id
    """
    
    df = spark.sql(sql2)
    
    display(df)
    
    
    
    
#     # Convert DATE Column from string to timestamp
#     # df_weather_2015_selected = df_weather_2015_selected.withColumn("DATE", to_timestamp(col("DATE"), "yyyy-MM-dd'T'HH:mm:ss"))
#     df_weather_selected = df_weather_selected.withColumn("DATE", df_weather_selected['DATE'].cast('timestamp'))
#     df_weather_selected = df_weather_selected.withColumn("DATE_PLUS_HOUR", df_weather_selected['DATE_PLUS_HOUR'].cast('timestamp'))
#     # Create new column with the hour of the weather reading (get rid of the minute reading)
#     df_weather_selected = df_weather_selected.withColumn('DATEHOUR', date_trunc("hour", df_weather_selected["DATE_PLUS_HOUR"]))
    
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
    
    
    sql = """
        SELECT _utc_date, _utc_hour_of_day, iata, AVG(ELEVATION) as Avg_Elevation,
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
        GROUP BY iata, _utc_date, _utc_hour_of_day
        ORDER BY iata, _utc_date, _utc_hour_of_day
    """
    df = spark.sql(sql)
    
    
    

stage_weather_data()

# COMMAND ----------

AIRPORTS_WS_LOC = f"{blob_url}/staged/airports_weatherstations"
df_airports_ws = spark.read.parquet(f"{AIRPORTS_WS_LOC}")
display(df_airports_ws)
# df_airports_ws.createOrReplaceTempView("airports_ws")

# COMMAND ----------


