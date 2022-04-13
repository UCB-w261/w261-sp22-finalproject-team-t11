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
from pyspark.sql.functions import when, to_timestamp, year, date_trunc, split, regexp_replace, array_max, length, substring, greatest

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

# MAGIC %md
# MAGIC ## Import Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather Data

# COMMAND ----------

display(dbutils.fs.ls(f"{blob_url}/raw/weather/_utc_year=2015"))

# COMMAND ----------

df_weather_2015 = spark.read.parquet(f"{blob_url}/raw/weather/_utc_year=2015/*.parquet")
df_weather_2015.createOrReplaceTempView("vw_weather_2015")
display(df_weather_2015)

# COMMAND ----------

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

# COMMAND ----------



# COMMAND ----------

df_weather_2015_selected = df_weather_2015.select(*cols_to_keep)
display(df_weather_2015_selected)
df_weather_2015.createOrReplaceTempView("vw_weather_2015_selected")

# COMMAND ----------

for field in df_weather_2015_selected.schema.fields:
    print(field.name +" , "+str(field.dataType))

# COMMAND ----------

# Convert DATE Column from string to timestamp
# df_weather_2015_selected = df_weather_2015_selected.withColumn("DATE", to_timestamp(col("DATE"), "yyyy-MM-dd'T'HH:mm:ss"))
df_weather_2015_selected = df_weather_2015_selected.withColumn("DATE", df_weather_2015_selected['DATE'].cast('timestamp'))
# Create new column with the hour of the weather reading (get rid of the minute reading)
df_weather_2015_selected = df_weather_2015_selected.withColumn('DATEHOUR', date_trunc("hour", df_weather_2015_selected["DATE"]))

# COMMAND ----------

# Convert Latitude, Longitude, Elevation from string to double
df_weather_2015_selected = df_weather_2015_selected.withColumn("LATITUDE",  df_weather_2015_selected['LATITUDE'].cast('double'))
df_weather_2015_selected = df_weather_2015_selected.withColumn("LONGITUDE",  df_weather_2015_selected['LONGITUDE'].cast('double'))
df_weather_2015_selected = df_weather_2015_selected.withColumn("ELEVATION",  df_weather_2015_selected['ELEVATION'].cast('double'))

# COMMAND ----------

# Apply transformations to convert weather conditions to appropriate values

# HOURLY ALTIMETER SETTING
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyAltimeterSetting', 
                                                               split(df_weather_2015_selected["HourlyAltimeterSetting"],'s')[0]\
                                                                     .cast('double'))

# DEW POINT TEMPERATURE
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyDewPointTemperature', 
                       split(df_weather_2015_selected["HourlyDewPointTemperature"],'s')[0])
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyDewPointTemperature',
                       df_weather_2015_selected["HourlyDewPointTemperature"].cast('double'))

# DRY BULB TEMPERATURE
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyDryBulbTemperature', 
                       split(df_weather_2015_selected["HourlyDryBulbTemperature"],'s')[0])
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyDryBulbTemperature',
                       df_weather_2015_selected["HourlyDryBulbTemperature"].cast('double'))

                                                               
df_weather_2015_selected.createOrReplaceTempView("vw_weather_2015_selected")

# COMMAND ----------

## Process PRECIPITATION
df_weather_2015_selected = df_weather_2015_selected.withColumn('Precip_Double', 
                                                               split(df_weather_2015_selected["HourlyPrecipitation"], 
                                                                     's')[0]) # get rid of s
df_weather_2015_selected = df_weather_2015_selected.withColumn('Precip_Double', 
                   regexp_replace('Precip_Double', 'T', '0')) # replace T with 0
df_weather_2015_selected = df_weather_2015_selected.withColumn('Precip_Double', 
                   regexp_replace('Precip_Double', '\*', '0')) # replace * with 0

# if there are multiple values in a single cell, get the maximum, cast the value as a double, replace NA values with 0
df_weather_2015_selected = df_weather_2015_selected.withColumn('Precip_Double', 
                                                               when(length(df_weather_2015_selected['Precip_Double']) % 4 == 0, 
                                                                    greatest(substring('Precip_Double', 1, 4), 
                                                                             substring('Precip_Double', 5, 4),
                                                                             substring('Precip_Double', 9, 4)))\
                                                               .when(length(df_weather_2015_selected['Precip_Double']) % 5 == 0,
                                                                     greatest(substring('Precip_Double', 1, 5), 
                                                                              substring('Precip_Double', 6, 5),
                                                                              substring('Precip_Double', 11, 5)))\
                                                               .otherwise(df_weather_2015_selected['Precip_Double'])\
                            .cast('double'))\
                            .na.fill(0.0, subset = ['Precip_Double'])
df_weather_2015_selected = df_weather_2015_selected.withColumn('Trace_Rain', 
                                                               when(df_weather_2015_selected['HourlyPrecipitation'].contains('T'), 
                                                                    1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.withColumn('NonZero_Rain', 
                                                               when((df_weather_2015_selected['Trace_Rain'] == 1) | \
                                                                    (df_weather_2015_selected['Precip_Double'] > 0), 
                                                                    1).otherwise(0))

df_weather_2015_selected.createOrReplaceTempView("vw_weather_2015_selected")

# COMMAND ----------

# HOURLY PRESSURE CHANGE
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyPressureChange', 
                                                               df_weather_2015_selected['HourlyPressureChange'].cast('double'))

# HOURLY PRESSURE TENDENCY
# CONVERT TO 3 SEPARATE BINARY COLUMNS BASED ON CATEGORICAL VALUE
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyPressureTendency', 
                                                               df_weather_2015_selected['HourlyPressureChange'].cast('double'))
    
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyPressureTendency_Increasing',
                                                               when(df_weather_2015_selected['HourlyPressureTendency'] <= 3,
                                                                    1)\
                                                               .otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyPressureTendency_Decreasing',
                                                               when(df_weather_2015_selected['HourlyPressureTendency'] >= 5,
                                                                    1)\
                                                               .otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyPressureTendency_Constant',
                                                               when(df_weather_2015_selected['HourlyPressureTendency'] == 4,
                                                                    1)\
                                                               .otherwise(0))

# HOURLY RELATIVE HUMIDITY
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyRelativeHumidity', 
                       split(df_weather_2015_selected["HourlyRelativeHumidity"],'s')[0])
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyRelativeHumidity',
                       df_weather_2015_selected["HourlyRelativeHumidity"].cast('double'))

# HOURLY SEA LEVEL PRESSURE
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlySeaLevelPressure', 
                       split(df_weather_2015_selected["HourlySeaLevelPressure"],'s')[0])
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlySeaLevelPressure',
                       when(df_weather_2015_selected["HourlySeaLevelPressure"] == '*', None)\
                       .otherwise(df_weather_2015_selected["HourlySeaLevelPressure"])\
                       .cast('double'))

# HOURLY STATION PRESSURE
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyStationPressure', 
                       split(df_weather_2015_selected["HourlyStationPressure"],'s')[0])
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyStationPressure',
                       df_weather_2015_selected["HourlyStationPressure"].cast('double'))

# HOURLY HORIZONTAL VISIBILITY
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyVisibility', 
                       split(df_weather_2015_selected["HourlyVisibility"],'s')[0])
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyVisibility', 
                       split(df_weather_2015_selected["HourlyVisibility"],'V')[0])
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyVisibility',
                       df_weather_2015_selected["HourlyVisibility"].cast('double'))

# WET BULB TEMPERATURE
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyWetBulbTemperature', 
                       split(df_weather_2015_selected["HourlyWetBulbTemperature"],'s')[0])
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyWetBulbTemperature',
                       df_weather_2015_selected["HourlyWetBulbTemperature"].cast('double'))

df_weather_2015_selected.createOrReplaceTempView("vw_weather_2015_selected")

# COMMAND ----------

# WIND DIRECTION
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyWindDirection', 
                       split(df_weather_2015_selected["HourlyWindDirection"],'s')[0])
# add new binary variable for calm winds, when hourly wind direction = 000
df_weather_2015_selected = df_weather_2015_selected.withColumn('Calm_Winds',
                       when(df_weather_2015_selected["HourlyWindDirection"] == '000', 1)\
                       .otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyWindDirection',
                       when(df_weather_2015_selected["HourlyWindDirection"] == 'VRB', None)\
                       .otherwise(df_weather_2015_selected["HourlyWindDirection"])\
                       .cast('double'))

# WIND GUST SPEED
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyWindGustSpeed', 
                       split(df_weather_2015_selected["HourlyWindGustSpeed"],'s')[0])
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyWindGustSpeed',
                       df_weather_2015_selected["HourlyWindGustSpeed"].cast('double'))


# HOURLY WIND SPEED
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyWindSpeed', 
                       split(df_weather_2015_selected["HourlyWindSpeed"],'s')[0])
df_weather_2015_selected = df_weather_2015_selected.withColumn('HourlyWindSpeed',
                       df_weather_2015_selected["HourlyWindSpeed"].cast('double'))

df_weather_2015_selected.createOrReplaceTempView("vw_weather_2015_selected")

# COMMAND ----------

# HOURLY SKY CONDITIONS
df_weather_2015_selected = df_weather_2015_selected.withColumn('Sky_Conditions_CLR', 
                                                               when(df_weather_2015_selected['HourlySkyConditions'].contains('CLR'), 
                                                                    1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.withColumn('Sky_Conditions_FEW', 
                                                               when(df_weather_2015_selected['HourlySkyConditions'].contains('FEW'), 
                                                                    1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.withColumn('Sky_Conditions_SCT', 
                                                               when(df_weather_2015_selected['HourlySkyConditions'].contains('SCT'), 
                                                                    1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.withColumn('Sky_Conditions_BKN', 
                                                               when(df_weather_2015_selected['HourlySkyConditions'].contains('BKN'), 
                                                                    1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.withColumn('Sky_Conditions_OVC', 
                                                               when(df_weather_2015_selected['HourlySkyConditions'].contains('OVC'), 
                                                                    1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.withColumn('Sky_Conditions_VV', 
                                                               when(df_weather_2015_selected['HourlySkyConditions'].contains('VV'), 
                                                                    1).otherwise(0))

# PRESENT WEATHER
df_weather_2015_selected = df_weather_2015_selected.\
                           withColumn('Present_Weather_Drizzle', 
                                      when(df_weather_2015_selected['HourlyPresentWeatherType'].contains('DZ'), 
                                           1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.\
                           withColumn('Present_Weather_Rain', 
                                      when(df_weather_2015_selected['HourlyPresentWeatherType'].contains('RA'), 
                                           1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.\
                           withColumn('Present_Weather_Snow', 
                                      when(df_weather_2015_selected['HourlyPresentWeatherType'].contains('SN'), 
                                           1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.\
                           withColumn('Present_Weather_SnowGrains', 
                                      when(df_weather_2015_selected['HourlyPresentWeatherType'].contains('SG'), 
                                           1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.\
                           withColumn('Present_Weather_IceCrystals', 
                                      when(df_weather_2015_selected['HourlyPresentWeatherType'].contains('IC'), 
                                           1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.\
                           withColumn('Present_Weather_Hail', 
                                      when((df_weather_2015_selected['HourlyPresentWeatherType'].contains('PL')) | 
                                           (df_weather_2015_selected['HourlyPresentWeatherType'].contains('GR')) |
                                           (df_weather_2015_selected['HourlyPresentWeatherType'].contains('GS')) |
                                           (df_weather_2015_selected['HourlyPresentWeatherType'].contains('HAIL')) |
                                           (df_weather_2015_selected['HourlyPresentWeatherType'].contains('|27')) |
                                           (df_weather_2015_selected['HourlyPresentWeatherType'].contains('SH:')), 
                                           1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.\
                           withColumn('Present_Weather_Mist', 
                                      when(df_weather_2015_selected['HourlyPresentWeatherType'].contains('BR'), 
                                           1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.\
                           withColumn('Present_Weather_Fog', 
                                      when(df_weather_2015_selected['HourlyPresentWeatherType'].contains('FG'), 
                                           1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.\
                           withColumn('Present_Weather_Smoke', 
                                      when(df_weather_2015_selected['HourlyPresentWeatherType'].contains('FU'), 
                                           1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.\
                           withColumn('Present_Weather_Dust', 
                                      when((df_weather_2015_selected['HourlyPresentWeatherType'].contains('FU')) |
                                           (df_weather_2015_selected['HourlyPresentWeatherType'].contains('VA')) |
                                           (df_weather_2015_selected['HourlyPresentWeatherType'].contains('DU')) |
                                           (df_weather_2015_selected['HourlyPresentWeatherType'].contains('SA')) |
                                           (df_weather_2015_selected['HourlyPresentWeatherType'].contains('PO')) |
                                           (df_weather_2015_selected['HourlyPresentWeatherType'].contains('PY')) |
                                           (df_weather_2015_selected['HourlyPresentWeatherType'].contains('SS')) |
                                           (df_weather_2015_selected['HourlyPresentWeatherType'].contains('DS')), 
                                           1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.\
                           withColumn('Present_Weather_Haze', 
                                      when(df_weather_2015_selected['HourlyPresentWeatherType'].contains('HZ'), 
                                           1).otherwise(0))
df_weather_2015_selected = df_weather_2015_selected.\
                           withColumn('Present_Weather_Storm', 
                                      when((df_weather_2015_selected['HourlyPresentWeatherType'].contains('SQ')) |
                                           (df_weather_2015_selected['HourlyPresentWeatherType'].contains('FC')) | 
                                           (df_weather_2015_selected['HourlyPresentWeatherType'].contains('TS')),
                                           1).otherwise(0))

df_weather_2015_selected.createOrReplaceTempView("vw_weather_2015_selected")

# COMMAND ----------

# DROP COLUMNS THAT WE'RE NO LONGER USING
drop_cols = ('HourlySkyConditions', 'HourlyPresentWeatherType', 'HourlyPrecipitation', 'HourlyPressureTendency')
df_weather_2015_selected = df_weather_2015_selected.drop(*drop_cols)

# COMMAND ----------

df_weather_2015_selected.createOrReplaceTempView("vw_weather_2015_selected")

# COMMAND ----------

display(df_weather_2015_selected)

# COMMAND ----------

# SCRATCH WORK

temp = df_weather_2015_selected.withColumn("HourlyPrecipitation", 
                                           df_weather_2015_selected['HourlyPrecipitation'].cast('double'))
temp.createOrReplaceTempView("temp")
display(temp)
#TimestampType()

# COMMAND ----------

# SCRATCH WORK
display(df_weather_2015_selected.withColumn("DATETIME", to_timestamp(col("DATE"), "yyyy-MM-dd'T'HH:00:00")))

# COMMAND ----------

# PRECIPITATION SCRATCH WORK

def count_nulls():
    sql = """
        SELECT STATION, DATE, HourlyPrecipitation
        FROM vw_weather_2015_selected
        WHERE HourlyPrecipitation = '0.01Ts'
        OR HourlyPrecipitation = 'TT'
        OR HourlyPrecipitation = 'Ts'
        OR HourlyPrecipitation > '0.00'
    """
    df = spark.sql(sql)
    df = df.withColumn('Precip_Double', split(df["HourlyPrecipitation"], 's')[0]) # get rid of s
    df = df.withColumn('Precip_Double', 
                       regexp_replace('Precip_Double', 'T', '0')) # replace T with 0
    df = df.withColumn('Precip_Double', 
                       regexp_replace('Precip_Double', '\*', '0')) # replace * with 0
    
    df = df.withColumn('Precip_Double', when(length(df['Precip_Double']) % 4 == 0, 
                                                    greatest(substring('Precip_Double', 1, 4), 
                                                             substring('Precip_Double', 5, 4),
                                                             substring('Precip_Double', 9, 4)))\
                       .when(length(df['Precip_Double']) % 5 == 0, 
                                                    greatest(substring('Precip_Double', 1, 5), 
                                                             substring('Precip_Double', 6, 5),
                                                             substring('Precip_Double', 11, 5)))\
                      .otherwise(df['Precip_Double']).cast('double')).na.fill(0.0, subset = ['Precip_Double'])
    
    df = df.withColumn('Trace_Rain', when(df['HourlyPrecipitation'].contains('T'), 1).otherwise(0))
    df = df.withColumn('NonZero_Rain', when((df['Trace_Rain'] == 1) | (df['Precip_Double'] > 0), 1).otherwise(0))
    
#     df = df.withColumn('Precip_Double', array_max(split(df["Precip_Double"], '\.')).cast('double')/100)
#     df = df.withColumn('Precip_Double', split(df["Precip_Double"], '\.')).cast('double')/100)
#     df = df.fillna(0, ['Precip_Double'])
#     df = df.withColumn('Precip_Double', 
#                        regexp_replace('Precip_Double', 'T', '0'))
#     df = df.withColumn('Precip_Double', 
#                        regexp_replace('Precip_Double', '\*', '0').cast('double'))\
#                        .na.fill(value = 0.0, subset = ['Precip_Double'])
    #.cast('double')
#     df = df.withColumn('Precip Double', regexp_replace('Precip Double', 'null', '0'))
#     df = df.na.fill(0, subset = ['HourlyPrecipitation'])
#     df = df.fillna('0', ['HourlyPrecipitation'])
    df = df.withColumn('DATEHOUR', date_trunc("hour", df["DATE"]))
    df.createOrReplaceTempView("temp2")
    
    sql2 = """
        SELECT STATION, DATEHOUR, AVG(Precip_Double)
        FROM temp2
        GROUP BY DATEHOUR, STATION
        ORDER BY DATEHOUR, STATION 
    """
    
#     df = spark.sql(sql2)
    
    display(df)
#     display(df.withColumn('Precip Double', \
#                           df.replace(['T', 'null'], ['0', '0'], 'Precip Double')['Precip Double'] \
#                           .cast('double')).withColumn('DATEHOUR', date_trunc("hour", df["DATE"])))

#         WHERE STATION = '72439754831'
#         AND DATE < '2015-01-10T00:56:00'
#         WHERE HourlyPrecipitation = '0.020.05s'
#         WHERE HourlyPrecipitation = '0.020.05s'
#         OR HourlyPrecipitation = '0.080.08'
#         OR HourlyPrecipitation = '0.020.01s'

count_nulls()

# COMMAND ----------

# SCRATCH WORK
def dew_point_scratchwork():
    sql = """
        SELECT STATION, DATE, HourlyDewPointTemperature
        FROM vw_weather_2015_selected
    """
    df = spark.sql(sql)
    
    df = df.withColumn('HourlyDewPointTemperature', 
                       split(df["HourlyDewPointTemperature"],'s')[0])
    df = df.withColumn('HourlyDewPointTemperature',
                       when(df["HourlyDewPointTemperature"] == '*', None).otherwise(df["HourlyDewPointTemperature"])\
                       .cast('double'))
    
    df.createOrReplaceTempView("temp2")
    
    sql2 = """
        SELECT HourlyDewPointTemperature, COUNT(*)
        FROM temp2
        GROUP BY HourlyDewPointTemperature
        ORDER BY HourlyDewPointTemperature
    """
    
    df = spark.sql(sql2)
    display(df)
    
dew_point_scratchwork()

# COMMAND ----------

# SCRATCH WORK
def dry_temp_scratchwork():
    sql = """
        SELECT STATION, DATE, HourlyVisibility
        FROM vw_weather_2015_selected
    """
    df = spark.sql(sql)
    
    df = df.withColumn('HourlyVisibility', 
                       split(df["HourlyVisibility"],'s')[0])
    df = df.withColumn('HourlyVisibility', 
                       split(df["HourlyVisibility"],'V')[0])
    df = df.withColumn('HourlyVisibility',
                       df["HourlyVisibility"].cast('double'))
    
    df.createOrReplaceTempView("temp2")
    
    sql2 = """
        SELECT HourlyVisibility, COUNT(*)
        FROM temp2
        GROUP BY HourlyVisibility
        ORDER BY HourlyVisibility
    """
    
    df = spark.sql(sql2)
    display(df)
    
dry_temp_scratchwork()

# COMMAND ----------

# PRESSURE TENDENCY SCRATCH WORK

def pressure_tendency_scratchwork():
    sql = """
        SELECT STATION, DATE, HourlyPressureTendency
        FROM vw_weather_2015_selected
    """
    df = spark.sql(sql)
    
    df = df.withColumn('HourlyPressureTendency', 
                       df['HourlyPressureTendency'].cast('double'))
    
    df = df.withColumn('HourlyPressureTendency_Increasing',
                       when(df['HourlyPressureTendency'] <= 3,
                            1)\
                       .otherwise(0))
    df = df.withColumn('HourlyPressureTendency_Decreasing',
                       when(df['HourlyPressureTendency'] >= 5,
                            1)\
                       .otherwise(0))
    df = df.withColumn('HourlyPressureTendency_Constant',
                       when(df['HourlyPressureTendency'] == 4,
                            1)\
                       .otherwise(0))
    
    df.createOrReplaceTempView("temp2")
    
    sql2 = """
        SELECT HourlyPressureTendency, HourlyPressureTendency_Increasing, HourlyPressureTendency_Decreasing, HourlyPressureTendency_Constant
        FROM temp2
    """
    
    df = spark.sql(sql2)
    display(df)
    
pressure_tendency_scratchwork()

# COMMAND ----------

# SCRATCH WORK
def count_distinct():
    sql = """
        SELECT HourlyPresentWeatherType, COUNT(*)
        FROM vw_weather_2015_selected
        GROUP BY HourlyPresentWeatherType
        ORDER BY HourlyPresentWeatherType
    """
    df = spark.sql(sql)
    display(df)

count_distinct()

# COMMAND ----------

# SCRATCH WORK
display(df_weather_2015_selected.filter((df_weather_2015_selected.STATION == '72439754831') & 
                                        (df_weather_2015_selected.DATE < '2015-01-01T06:30:00')))

# COMMAND ----------

airports_ws = spark.read.parquet(f"{blob_url}/staged/airports_weatherstations")
display(airports_ws)

# COMMAND ----------


