# Databricks notebook source
# MAGIC %md
# MAGIC # EDA on Weather Datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

from pyspark.sql.functions import col, countDistinct
# from pyspark.sql import SparkSession ? --> do we need this?
# do we need to build a Spark Context?
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pyspark
import datetime as dt
from pyspark.sql.functions import when

from configuration_v01 import Configuration
configuration = Configuration()

WEATHER_LOC = f"{configuration.blob_url}/staged/weather"


# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Weather Data

# COMMAND ----------

# DBTITLE 1,LCD Data NEW
df_weather = spark.read.parquet(WEATHER_LOC + '/clean_weather_data.parquet')
df_weather.createOrReplaceTempView("weather_vw")
display(df_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stations Data

# COMMAND ----------

df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
df_stations.createOrReplaceTempView("stations_vw")
display(df_stations)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join Weather Data with Ram's list of Stations

# COMMAND ----------

#### OLD DATASET ####
weather_stations_use = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/jconde@berkeley.edu/weather_stations_use.csv")
display(weather_stations_use)
weather_stations_use.createOrReplaceTempView("weather_stations_use")

# COMMAND ----------

closest_stations = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/jconde@berkeley.edu/closest_ws_stations.csv")
display(closest_stations)
closest_stations.createOrReplaceTempView("closest_stations")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC wnd, cig, vis, tmp, dew, slp
# MAGIC 
# MAGIC preciptation: AA1-AA4
# MAGIC 
# MAGIC snow: AJ1 -- had a lot of missing data, not including this column
# MAGIC 
# MAGIC daily weather occurrence: AT1-AT8

# COMMAND ----------

def selected_stations_dailyweather(table):
    sql = """
        SELECT station, date, source, name, AJ1
        FROM """ + table + """
        RIGHT JOIN weather_stations_use ON """ + table + """.station = weather_stations_use.ws_id
        """
    
    result_df = spark.sql(sql)
    return result_df
dailyweather_filtered = selected_stations_dailyweather('weather_vw')
display(dailyweather_filtered)
# weather_filtered.createOrReplaceTempView("weather_selected_vw")

# COMMAND ----------

def selected_stations(table):
    sql = """
        SELECT station, date, source, latitude, longitude, elevation, name, report_type, call_sign, quality_control, wnd, cig, vis, tmp, dew, slp, AA1, AA2, AA3, AA4, AT1, AT2, AT3, AT4, AT5, AT6, AT7, AT8
        FROM """ + table + """
        RIGHT JOIN closest_stations ON """ + table + """.station = closest_stations.ws_id
        """
    
    result_df = spark.sql(sql)
    return result_df
weather_filtered = selected_stations('weather_vw')
display(weather_filtered)
weather_filtered.createOrReplaceTempView("weather_selected_vw")

# COMMAND ----------

# MAGIC %md
# MAGIC ## [OLD] Weather EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ### [OLD] Select relevant columns from big weather table

# COMMAND ----------

us_weather_new.createOrReplaceTempView("us_weather_vw")

# COMMAND ----------

# MAGIC %md
# MAGIC ### [OLD] Clean columns

# COMMAND ----------

wind_split = pyspark.sql.functions.split(us_weather['WND'], ',')
us_weather_new = us_weather.withColumn('WND_DirectionAngle', wind_split.getItem(0))
us_weather_new = us_weather_new.withColumn('WND_DirectionQuality', wind_split.getItem(1).cast("float"))
us_weather_new = us_weather_new.withColumn('WND_Type', wind_split.getItem(2))
us_weather_new = us_weather_new.withColumn('WND_Speed', wind_split.getItem(3).cast("float")/10)
us_weather_new = us_weather_new.withColumn('WND_SpeedQuality', wind_split.getItem(4))
us_weather_new = us_weather_new.drop('WND')

cig_split = pyspark.sql.functions.split(us_weather['CIG'], ',')
us_weather_new = us_weather_new.withColumn('CIG_CeilingHeightDim', cig_split.getItem(0).cast("float"))
us_weather_new = us_weather_new.withColumn('CIG_CeilingQuality', cig_split.getItem(1))
us_weather_new = us_weather_new.withColumn('CIG_CeilingDetermination', cig_split.getItem(2))
us_weather_new = us_weather_new.withColumn('CIG_CeilingAndVisibility', cig_split.getItem(3))
us_weather_new = us_weather_new.drop('CIG')

vis_split = pyspark.sql.functions.split(us_weather['VIS'], ',')
us_weather_new = us_weather_new.withColumn('VIS_Horizontal', vis_split.getItem(0).cast("float"))
us_weather_new = us_weather_new.withColumn('VIS_DistanceQuality', vis_split.getItem(1))
us_weather_new = us_weather_new.withColumn('VIS_Variability', vis_split.getItem(2))
us_weather_new = us_weather_new.withColumn('VIS_QualityOfVariability', vis_split.getItem(3))
us_weather_new = us_weather_new.drop('VIS')

tmp_split = pyspark.sql.functions.split(us_weather['TMP'], ',')
us_weather_new = us_weather_new.withColumn('TMP_tmp', tmp_split.getItem(0).cast("float")/10)
us_weather_new = us_weather_new.withColumn('TMP_Quality', tmp_split.getItem(1))
us_weather_new = us_weather_new.drop('TMP')

dew_split = pyspark.sql.functions.split(us_weather['DEW'], ',')
us_weather_new = us_weather_new.withColumn('DEW_dew', dew_split.getItem(0).cast("float")/10)
us_weather_new = us_weather_new.withColumn('DEW_Quality', dew_split.getItem(1))
us_weather_new = us_weather_new.drop('DEW')

slp_split = pyspark.sql.functions.split(us_weather['SLP'], ',')
us_weather_new = us_weather_new.withColumn('SLP_slp', slp_split.getItem(0).cast("float")/10)
us_weather_new = us_weather_new.withColumn('SLP_Quality', slp_split.getItem(1))
us_weather_new = us_weather_new.drop('SLP')

# COMMAND ----------

display(us_weather_new)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean Columns

# COMMAND ----------

# WIND COLUMN
# split the column and drop the original
wind_split = pyspark.sql.functions.split(weather_filtered['WND'], ',')
weather_filtered_new = weather_filtered.withColumn('WND_DirectionAngle', wind_split.getItem(0).cast("float"))
weather_filtered_new = weather_filtered_new.withColumn('WND_DirectionQuality', wind_split.getItem(1).cast("float"))
weather_filtered_new = weather_filtered_new.withColumn('WND_Type', wind_split.getItem(2))
weather_filtered_new = weather_filtered_new.withColumn('WND_Speed', wind_split.getItem(3).cast("float")/10)
weather_filtered_new = weather_filtered_new.withColumn('WND_SpeedQuality', wind_split.getItem(4))
weather_filtered_new = weather_filtered_new.drop('WND')

# CEILING HEIGHT
# split the column and drop the original
cig_split = pyspark.sql.functions.split(weather_filtered['CIG'], ',')
weather_filtered_new = weather_filtered_new.withColumn('CIG_CeilingHeightDim', cig_split.getItem(0).cast("float"))
weather_filtered_new = weather_filtered_new.withColumn('CIG_CeilingQuality', cig_split.getItem(1))
weather_filtered_new = weather_filtered_new.withColumn('CIG_CeilingDetermination', cig_split.getItem(2))
weather_filtered_new = weather_filtered_new.withColumn('CIG_CeilingAndVisibility', cig_split.getItem(3))
weather_filtered_new = weather_filtered_new.drop('CIG')

# HORIZONTAL VISIBILITY
# split the column and drop the original
vis_split = pyspark.sql.functions.split(weather_filtered['VIS'], ',')
weather_filtered_new = weather_filtered_new.withColumn('VIS_Horizontal', vis_split.getItem(0).cast("float"))
weather_filtered_new = weather_filtered_new.withColumn('VIS_DistanceQuality', vis_split.getItem(1))
weather_filtered_new = weather_filtered_new.withColumn('VIS_Variability', vis_split.getItem(2))
weather_filtered_new = weather_filtered_new.withColumn('VIS_QualityOfVariability', vis_split.getItem(3))
weather_filtered_new = weather_filtered_new.drop('VIS')

# TEMPERATURE
# split the column and drop the original
tmp_split = pyspark.sql.functions.split(weather_filtered['TMP'], ',')
weather_filtered_new = weather_filtered_new.withColumn('TMP_tmp', tmp_split.getItem(0).cast("float")/10)
weather_filtered_new = weather_filtered_new.withColumn('TMP_Quality', tmp_split.getItem(1))
weather_filtered_new = weather_filtered_new.drop('TMP')

# DEW POINT
# split the column and drop the original
dew_split = pyspark.sql.functions.split(weather_filtered['DEW'], ',')
weather_filtered_new = weather_filtered_new.withColumn('DEW_dew', dew_split.getItem(0).cast("float")/10)
weather_filtered_new = weather_filtered_new.withColumn('DEW_Quality', dew_split.getItem(1))
weather_filtered_new = weather_filtered_new.drop('DEW')

# AIR PRESSURE
# split the column and drop the original
slp_split = pyspark.sql.functions.split(weather_filtered['SLP'], ',')
weather_filtered_new = weather_filtered_new.withColumn('SLP_slp', slp_split.getItem(0).cast("float")/10)
weather_filtered_new = weather_filtered_new.withColumn('SLP_Quality', slp_split.getItem(1))
weather_filtered_new = weather_filtered_new.drop('SLP')

# PRECIPITATION
# get the precipitation in mm and the hours of measurement
# convert to mm/hr value (?)
# drop the original columns
precip_split = pyspark.sql.functions.split(weather_filtered['AA1'], ',')
weather_filtered_new = weather_filtered_new.withColumn('PRECIP_hrs', precip_split.getItem(0).cast("float"))
weather_filtered_new = weather_filtered_new.withColumn('PRECIP_mm', precip_split.getItem(1).cast("float")/10)
weather_filtered_new = weather_filtered_new.drop('AA1')
weather_filtered_new = weather_filtered_new.drop('AA2')
weather_filtered_new = weather_filtered_new.drop('AA3')
weather_filtered_new = weather_filtered_new.drop('AA4')
weather_filtered_new = weather_filtered_new.fillna( { 'PRECIP_hrs':1.0, 'PRECIP_mm':0.0 } )
weather_filtered_new = weather_filtered_new.withColumn('PRECIP_rate', 
                                                       weather_filtered_new.PRECIP_mm / weather_filtered_new.PRECIP_hrs)

# DAILY WEATHER CONDITIONS
# extract the conditions and create binary columns
daily_split_1 = pyspark.sql.functions.split(weather_filtered['AT1'], ',')
daily_split_2 = pyspark.sql.functions.split(weather_filtered['AT2'], ',')
daily_split_3 = pyspark.sql.functions.split(weather_filtered['AT3'], ',')
daily_split_4 = pyspark.sql.functions.split(weather_filtered['AT4'], ',')
weather_filtered_new = weather_filtered_new.withColumn('DAILY_1', daily_split_1.getItem(1))
weather_filtered_new = weather_filtered_new.withColumn('DAILY_2', daily_split_2.getItem(1))
weather_filtered_new = weather_filtered_new.withColumn('DAILY_3', daily_split_3.getItem(1))
weather_filtered_new = weather_filtered_new.withColumn('DAILY_4', daily_split_4.getItem(1))
weather_filtered_new = weather_filtered_new.drop('AT1')
weather_filtered_new = weather_filtered_new.drop('AT2')
weather_filtered_new = weather_filtered_new.drop('AT3')
weather_filtered_new = weather_filtered_new.drop('AT4')
weather_filtered_new = weather_filtered_new.drop('AT5')
weather_filtered_new = weather_filtered_new.drop('AT6')
weather_filtered_new = weather_filtered_new.drop('AT7')
weather_filtered_new = weather_filtered_new.drop('AT8')
weather_filtered_new = weather_filtered_new.withColumn('FOG',
                                                       when((col('DAILY_1') == '01') | 
                                                            (col('DAILY_2') == '01') | 
                                                            (col('DAILY_3') == '01') | 
                                                            (col('DAILY_4') == '01') |
                                                            (col('DAILY_1') == '02') | 
                                                            (col('DAILY_2') == '02') | 
                                                            (col('DAILY_3') == '02') | 
                                                            (col('DAILY_4') == '02') |
                                                            (col('DAILY_1') == '21') |
                                                            (col('DAILY_2') == '21') |
                                                            (col('DAILY_3') == '21') |
                                                            (col('DAILY_4') == '21') |
                                                            (col('DAILY_1') == '22') |
                                                            (col('DAILY_2') == '22') |
                                                            (col('DAILY_3') == '22') |
                                                            (col('DAILY_4') == '22'), 1).otherwise(0))
weather_filtered_new = weather_filtered_new.withColumn('THUNDER', 
                                                       when((col('DAILY_1') == '03') | 
                                                            (col('DAILY_2') == '03') | 
                                                            (col('DAILY_3') == '03') |
                                                            (col('DAILY_4') == '03'), 1).otherwise(0))
weather_filtered_new = weather_filtered_new.withColumn('ICE', 
                                                       when((col('DAILY_1') == '04') | 
                                                            (col('DAILY_2') == '04') | 
                                                            (col('DAILY_3') == '04') |
                                                            (col('DAILY_4') == '04') |
                                                            (col('DAILY_1') == '05') | 
                                                            (col('DAILY_2') == '05') | 
                                                            (col('DAILY_3') == '05') |
                                                            (col('DAILY_4') == '05'), 1).otherwise(0))
weather_filtered_new = weather_filtered_new.withColumn('GLAZE_RIME', 
                                                       when((col('DAILY_1') == '06') | 
                                                            (col('DAILY_2') == '06') | 
                                                            (col('DAILY_3') == '06') |
                                                            (col('DAILY_4') == '06'), 1).otherwise(0))
weather_filtered_new = weather_filtered_new.withColumn('DUST_SMOKE_HAZE', 
                                                       when((col('DAILY_1') == '07') | 
                                                            (col('DAILY_2') == '07') | 
                                                            (col('DAILY_3') == '07') |
                                                            (col('DAILY_4') == '07') |
                                                            (col('DAILY_1') == '08') | 
                                                            (col('DAILY_2') == '08') | 
                                                            (col('DAILY_3') == '08') |
                                                            (col('DAILY_4') == '08'), 1).otherwise(0))
weather_filtered_new = weather_filtered_new.withColumn('BLOWING_SPRAY_SNOW', 
                                                       when((col('DAILY_1') == '09') | 
                                                            (col('DAILY_2') == '09') | 
                                                            (col('DAILY_3') == '09') |
                                                            (col('DAILY_4') == '09') |
                                                            (col('DAILY_1') == '12') | 
                                                            (col('DAILY_2') == '12') | 
                                                            (col('DAILY_3') == '12') |
                                                            (col('DAILY_4') == '12'), 1).otherwise(0))
weather_filtered_new = weather_filtered_new.withColumn('TORNADO', 
                                                       when((col('DAILY_1') == '10') | 
                                                            (col('DAILY_2') == '10') | 
                                                            (col('DAILY_3') == '10') |
                                                            (col('DAILY_4') == '10'), 1).otherwise(0))
weather_filtered_new = weather_filtered_new.withColumn('HIGH_WINDS', 
                                                       when((col('DAILY_1') == '11') | 
                                                            (col('DAILY_2') == '11') | 
                                                            (col('DAILY_3') == '11') |
                                                            (col('DAILY_4') == '11'), 1).otherwise(0))
weather_filtered_new = weather_filtered_new.withColumn('MIST_DRIZZLE', 
                                                       when((col('DAILY_1') == '13') | 
                                                            (col('DAILY_2') == '13') | 
                                                            (col('DAILY_3') == '13') |
                                                            (col('DAILY_4') == '13') |
                                                            (col('DAILY_1') == '14') | 
                                                            (col('DAILY_2') == '14') | 
                                                            (col('DAILY_3') == '14') |
                                                            (col('DAILY_4') == '14'), 1).otherwise(0))
weather_filtered_new = weather_filtered_new.withColumn('RAIN', 
                                                       when((col('DAILY_1') == '16') | 
                                                            (col('DAILY_2') == '16') | 
                                                            (col('DAILY_3') == '16') |
                                                            (col('DAILY_4') == '16') |
                                                            (col('DAILY_1') == '19') | 
                                                            (col('DAILY_2') == '19') | 
                                                            (col('DAILY_3') == '19') |
                                                            (col('DAILY_4') == '19'), 1).otherwise(0))
weather_filtered_new = weather_filtered_new.withColumn('FREEZING_RAIN', 
                                                       when((col('DAILY_1') == '15') | 
                                                            (col('DAILY_2') == '15') | 
                                                            (col('DAILY_3') == '15') |
                                                            (col('DAILY_4') == '15') | 
                                                            (col('DAILY_1') == '17') | 
                                                            (col('DAILY_2') == '17') | 
                                                            (col('DAILY_3') == '17') |
                                                            (col('DAILY_4') == '17'), 1).otherwise(0))
weather_filtered_new = weather_filtered_new.withColumn('SNOW', 
                                                       when((col('DAILY_1') == '18') | 
                                                            (col('DAILY_2') == '18') | 
                                                            (col('DAILY_3') == '18') |
                                                            (col('DAILY_4') == '18'), 1).otherwise(0))
weather_filtered_new = weather_filtered_new.drop('DAILY_1')
weather_filtered_new = weather_filtered_new.drop('DAILY_2')
weather_filtered_new = weather_filtered_new.drop('DAILY_3')
weather_filtered_new = weather_filtered_new.drop('DAILY_4')

# COMMAND ----------

type(slp_split.getItem(0))

# COMMAND ----------

display(weather_filtered_new)
weather_filtered_new.createOrReplaceTempView("weather_filtered_vw")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checking Variable Quality

# COMMAND ----------

# MAGIC %md
# MAGIC #### Temperature

# COMMAND ----------

# Temperature Data Quality
def weather_locations(table):
    sql = """
        SELECT TMP_Quality, COUNT(TMP_Quality)
        FROM """ + table + """
        GROUP BY TMP_Quality
        ORDER BY TMP_Quality
        """
    
    result_df = spark.sql(sql)
    return result_df
us_weather = weather_locations('weather_filtered_vw')
display(us_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Dew Point

# COMMAND ----------

def dew_quality(table):
    sql = """
        SELECT DEW_Quality, COUNT(DEW_Quality)
        FROM """ + table + """
        GROUP BY DEW_Quality
        ORDER BY DEW_Quality
        """
    
    result_df = spark.sql(sql)
    return result_df
dew_counts = dew_quality('weather_filtered_vw')
display(dew_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Wind Direction

# COMMAND ----------

def wind_direction_quality(table):
    sql = """
        SELECT WND_DirectionQuality, COUNT(WND_DirectionQuality)
        FROM """ + table + """
        GROUP BY WND_DirectionQuality
        ORDER BY WND_DirectionQuality
        """
    
    result_df = spark.sql(sql)
    return result_df
wind_direction_quality_counts = wind_direction_quality('weather_filtered_vw')
display(wind_direction_quality_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Wind Speed

# COMMAND ----------

def wind_speed_quality(table):
    sql = """
        SELECT WND_SpeedQuality, COUNT(WND_SpeedQuality)
        FROM """ + table + """
        GROUP BY WND_SpeedQuality
        ORDER BY WND_SpeedQuality
        """
    
    result_df = spark.sql(sql)
    return result_df
wind_speed_quality_counts = wind_speed_quality('weather_filtered_vw')
display(wind_speed_quality_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Ceiling Quality

# COMMAND ----------

def ceiling_quality(table):
    sql = """
        SELECT CIG_CeilingQuality, COUNT(CIG_CeilingQuality)
        FROM """ + table + """
        GROUP BY CIG_CeilingQuality
        ORDER BY CIG_CeilingQuality
        """
    
    result_df = spark.sql(sql)
    return result_df
ceiling_quality_counts = ceiling_quality('weather_filtered_vw')
display(ceiling_quality_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visibility

# COMMAND ----------

def visibility_quality(table):
    sql = """
        SELECT VIS_DistanceQuality, COUNT(VIS_DistanceQuality)
        FROM """ + table + """
        GROUP BY VIS_DistanceQuality
        ORDER BY VIS_DistanceQuality
        """
    
    result_df = spark.sql(sql)
    return result_df
visibility_quality_counts = visibility_quality('weather_filtered_vw')
display(visibility_quality_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pressure

# COMMAND ----------

def pressure_quality(table):
    sql = """
        SELECT SLP_Quality, COUNT(SLP_Quality)
        FROM """ + table + """
        GROUP BY SLP_Quality
        ORDER BY SLP_Quality
        """
    
    result_df = spark.sql(sql)
    return result_df
pressure_quality_counts = pressure_quality('weather_filtered_vw')
display(pressure_quality_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing Values

# COMMAND ----------

# MAGIC %md
# MAGIC #### Temperature

# COMMAND ----------

def temp_missing(table):
    sql = """
        SELECT TMP_tmp, COUNT(TMP_tmp)
        FROM """ + table + """
        WHERE TMP_tmp > 800
        GROUP BY TMP_tmp
        """
    
    result_df = spark.sql(sql)
    return result_df
temp_missing_counts = temp_missing('weather_filtered_vw')
display(temp_missing_counts)

# COMMAND ----------

def temp_missing(table):
    sql = """
        SELECT name, COUNT(TMP_tmp)
        FROM """ + table + """
        WHERE TMP_tmp > 800
        GROUP BY name 
        """
    
    result_df = spark.sql(sql)
    return result_df
temp_missing_counts_byname = temp_missing('weather_filtered_vw')
display(temp_missing_counts_byname)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Wind Direction

# COMMAND ----------

def wind_direction_missing(table):
    sql = """
        SELECT WND_DirectionAngle, COUNT(WND_DirectionAngle)
        FROM """ + table + """
        WHERE WND_DirectionAngle = '999'
        GROUP BY WND_DirectionAngle
        """
    
    result_df = spark.sql(sql)
    return result_df
wind_direction_missing_counts = wind_direction_missing('weather_filtered_vw')
display(wind_direction_missing_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Other Weather EDA

# COMMAND ----------

# MAGIC %md
# MAGIC #### Temp Min/Max

# COMMAND ----------

def temp_range(table):
    sql = """
        SELECT MAX(TMP_tmp), MIN(TMP_tmp)
        FROM """ + table + """
        WHERE TMP_tmp < 999
        """
    
    result_df = spark.sql(sql)
    return result_df
temp_range_vals = temp_range('weather_filtered_vw')
display(temp_range_vals)

# COMMAND ----------

weather_filtered_new.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Daily Weather Condition Averages

# COMMAND ----------

def daily_averages(table):
    sql = """
        SELECT AVG(FOG), AVG(THUNDER), AVG(ICE), AVG(GLAZE_RIME), AVG(DUST_SMOKE_HAZE), AVG(BLOWING_SPRAY_SNOW), 
        AVG(TORNADO), AVG(HIGH_WINDS), AVG(MIST_DRIZZLE), AVG(RAIN), AVG(FREEZING_RAIN), AVG(SNOW)
        FROM """ + table + """
        """
    
    result_df = spark.sql(sql)
    return result_df
daily_avgs = daily_averages('weather_filtered_vw')
display(daily_avgs)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualizing with a Sample

# COMMAND ----------

sampled_weather = weather_filtered_new.sample(fraction = 0.05)
sampled_weather_pd = sampled_weather.toPandas()
sampled_weather_pd

# COMMAND ----------

# MAGIC %md
# MAGIC #### Temperature

# COMMAND ----------

sns.boxplot(x = sampled_weather_pd[sampled_weather_pd['TMP_tmp'] < 800]['date'].dt.month, 
            y = sampled_weather_pd[sampled_weather_pd['TMP_tmp'] < 800]['TMP_tmp']);
plt.xlabel('Month');
plt.ylabel('Temperature (°C)');
plt.title('Temperature by Month (Missing Data Removed)');

# COMMAND ----------

sns.kdeplot(x = sampled_weather_pd[sampled_weather_pd['TMP_tmp'] < 800]['TMP_tmp'], 
            hue = sampled_weather_pd[sampled_weather_pd['TMP_tmp'] < 800]['date'].dt.month);
# plt.xlabel('Month');
plt.xlabel('Temperature (°C)');
plt.title('Temperature by Month (Missing Data Removed)');

# COMMAND ----------

# MAGIC %md
# MAGIC #### Wind Direction Angle

# COMMAND ----------

sns.boxplot(x = sampled_weather_pd[sampled_weather_pd['WND_DirectionAngle'] < 800]['date'].dt.month, 
            y = sampled_weather_pd[sampled_weather_pd['WND_DirectionAngle'] < 800]['WND_DirectionAngle']);
plt.xlabel('Month');
plt.ylabel('Wind Direction Angle');
plt.title('Wind Direction Angle by Month (Missing Data Removed)');

# COMMAND ----------

sns.histplot(x = sampled_weather_pd[sampled_weather_pd['WND_DirectionAngle'] < 800]['WND_DirectionAngle'], kde = True, stat = 'density');
plt.xlabel('Wind Direction Angle');
plt.ylabel('Density');
plt.title('Wind Direction Angle Distribution (Missing Data Removed)');

# COMMAND ----------

# MAGIC %md
# MAGIC #### Wind Speed

# COMMAND ----------

sns.histplot(x = sampled_weather_pd[sampled_weather_pd['WND_Speed'] < 800]['WND_Speed'], kde = True, stat = 'density');
plt.xlabel('Wind Speed');
plt.ylabel('Density');
plt.title('Wind Speed Distribution (Missing Data Removed)');

# COMMAND ----------

sns.boxplot(x = sampled_weather_pd[sampled_weather_pd['WND_Speed'] < 800]['date'].dt.month, 
            y = sampled_weather_pd[sampled_weather_pd['WND_Speed'] < 800]['WND_Speed']);
plt.xlabel('Month');
plt.ylabel('Wind Speed');
plt.title('Wind Speed by Month (Missing Data Removed)');

# COMMAND ----------

sns.kdeplot(x = sampled_weather_pd[sampled_weather_pd['WND_Speed'] < 800]['WND_Speed'], 
            hue = sampled_weather_pd[sampled_weather_pd['WND_Speed'] < 800]['date'].dt.month);
# plt.xlabel('Month');
plt.xlabel('Wind Speed (m/s)');
plt.title('Wind Speed by Month (Missing Data Removed)');

# COMMAND ----------

sns.boxplot(x = sampled_weather_pd[sampled_weather_pd['WND_Speed'] < 800]['date'].dt.hour, 
            y = sampled_weather_pd[sampled_weather_pd['WND_Speed'] < 800]['WND_Speed']);
plt.xlabel('Hour of Day');
plt.ylabel('Wind Speed');
plt.title('Wind Speed by Hour (Missing Data Removed)');

# COMMAND ----------

# MAGIC %md
# MAGIC #### Ceiling Height

# COMMAND ----------

sns.histplot(x = sampled_weather_pd[sampled_weather_pd['CIG_CeilingHeightDim'] < 88888]['CIG_CeilingHeightDim'], 
             kde = True, stat = 'density');
plt.xlabel('Ceiling Height');
plt.ylabel('Density');
plt.title('Ceiling Height Distribution (Missing Data Removed)');

# COMMAND ----------

sns.kdeplot(x = sampled_weather_pd[sampled_weather_pd['CIG_CeilingHeightDim'] < 88888]['CIG_CeilingHeightDim'], 
            hue = sampled_weather_pd[sampled_weather_pd['CIG_CeilingHeightDim'] < 88888]['date'].dt.month);
# plt.xlabel('Month');
plt.xlabel('Ceiling Height');
plt.title('Ceiling Height by Month (Missing Data Removed)');

# COMMAND ----------

sns.boxplot(x = sampled_weather_pd[sampled_weather_pd['VIS_Horizontal'] < 88888]['date'].dt.hour, 
            y = sampled_weather_pd[sampled_weather_pd['VIS_Horizontal'] < 88888]['VIS_Horizontal']);
plt.xlabel('Hour of Day');
plt.ylabel('Horizontal Visibility');
plt.title('Horizontal Visibility by Hour (Missing Data Removed)');

# COMMAND ----------

def test():
    df = spark.read.parquet(FINAL_JOINED_DATA_ALL)
    display(df)
    print(f'Total number of flights: {df.count()}')
    display(df.filter((col('FL_DATE') == '2015-01-01') & (col('ORIGIN') == 'SFO')))
    
test()

# COMMAND ----------

# MAGIC %md
# MAGIC ## NEW Weather EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic EDA

# COMMAND ----------

def feature_min_max(feature):
    sql = """
        SELECT MIN(""" + feature + """), MAX(""" + feature + """)
        from weather_vw
    """
    
    result_df = spark.sql(sql)
    return result_df
    
display(feature_min_max('Avg_HourlyWetBulbTemperature'))

# COMMAND ----------

display(feature_min_max('Avg_HourlyWetBulbTemperature'))

# COMMAND ----------

display(feature_min_max('Avg_HourlyDryBulbTemperature'))

# COMMAND ----------

display(feature_min_max('Avg_HourlyWindGustSpeed'))

# COMMAND ----------

display(feature_min_max('Avg_HourlyWindSpeed'))

# COMMAND ----------

def feature_avg(feature):
    sql = """
        SELECT AVG(""" + feature + """)
        from weather_vw
    """
    
    result_df = spark.sql(sql)
    return result_df
    
display(feature_min_max('Avg_HourlyWetBulbTemperature'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample Data for Viz

# COMMAND ----------

sampled_weather = df_weather.sample(fraction = 0.025)
sampled_weather_pd = sampled_weather.toPandas()
sampled_weather_pd

# COMMAND ----------

sns.boxplot(x = sampled_weather_pd['DATEHOUR'].dt.month, 
            y = sampled_weather_pd['Avg_HourlyDryBulbTemperature']);
plt.xlabel('Month');
plt.ylabel('Temperature (°F)');
plt.title('Temperature by Month (NaNs Removed)');

# COMMAND ----------

temp_by_month = sns.kdeplot(x = sampled_weather_pd['Avg_HourlyDryBulbTemperature'], 
                            hue = sampled_weather_pd['DATEHOUR'].dt.month,
                           legend = False);
plt.xlabel('Temperature (°F)');
plt.legend(title = "Month", labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
plt.title('Temperature by Month (Missing Data Removed)');

# COMMAND ----------

sns.histplot(x = sampled_weather_pd['Avg_HourlyWindSpeed'], kde = True, stat = 'density');
plt.xlabel('Wind Speed');
plt.ylabel('Density');
plt.title('Wind Speed Distribution (Missing Data Removed)');

# COMMAND ----------

sns.kdeplot(x = sampled_weather_pd[sampled_weather_pd['Avg_HourlyWindSpeed'] < 150]['Avg_HourlyWindSpeed'], 
            hue = sampled_weather_pd[sampled_weather_pd['Avg_HourlyWindSpeed'] < 150]['DATEHOUR'].dt.month);
# plt.xlabel('Month');
plt.xlabel('Wind Speed (m/s)');
plt.title('Wind Speed by Month (Missing Data Removed)');

# COMMAND ----------

sns.boxplot(x = sampled_weather_pd[sampled_weather_pd['Avg_HourlyWindSpeed'] < 150]['DATEHOUR'].dt.hour, 
            y = sampled_weather_pd[sampled_weather_pd['Avg_HourlyWindSpeed'] < 150]['Avg_HourlyWindSpeed']);
plt.xlabel('Hour of Day');
plt.ylabel('Wind Speed');
plt.title('Wind Speed by Hour (Missing Data Removed)');

# COMMAND ----------

sampled_weather_pd['Present_Weather_Drizzle'].value_counts().index

# COMMAND ----------

print(sampled_weather_pd['Present_Weather_Drizzle'].value_counts())
fig, ax = plt.subplots()
plt.bar(sampled_weather_pd['Present_Weather_Drizzle'].value_counts().index, height = sampled_weather_pd['Present_Weather_Drizzle'].value_counts()/sum(sampled_weather_pd['Present_Weather_Drizzle'].value_counts()));
plt.xlabel('Dizzle (1 = Yes, 0 = No)');
ax.set_xticks([0, 1])
plt.ylabel('Proportion');
plt.title('Drizzle');

# COMMAND ----------

fig, ax = plt.subplots()
plt.bar(sampled_weather_pd['Present_Weather_Rain'].value_counts().index, height = sampled_weather_pd['Present_Weather_Rain'].value_counts()/sum(sampled_weather_pd['Present_Weather_Rain'].value_counts()));
plt.xlabel('Rain (1 = Yes, 0 = No)');
ax.set_xticks([0, 1])
plt.ylabel('Proportion');
plt.title('Rain');

# COMMAND ----------

print(sampled_weather_pd['Present_Weather_Snow'].value_counts())
fig, ax = plt.subplots()
plt.bar(sampled_weather_pd['Present_Weather_Snow'].value_counts().index, height = sampled_weather_pd['Present_Weather_Snow'].value_counts()/sum(sampled_weather_pd['Present_Weather_Snow'].value_counts()));
plt.xlabel('Snow (1 = Yes, 0 = No)');
ax.set_xticks([0, 1])
plt.ylabel('Proportion');
plt.title('Snow');

# COMMAND ----------

fig, ax = plt.subplots()
print(sampled_weather_pd['Present_Weather_SnowGrains'].value_counts())
plt.bar(sampled_weather_pd['Present_Weather_SnowGrains'].value_counts().index, height = sampled_weather_pd['Present_Weather_SnowGrains'].value_counts()/sum(sampled_weather_pd['Present_Weather_SnowGrains'].value_counts()));
plt.xlabel('SnowGrains (1 = Yes, 0 = No)');
ax.set_xticks([0, 1])
plt.ylabel('Proportion');
plt.title('SnowGrains');

# COMMAND ----------

fig, ax = plt.subplots()
print(sampled_weather_pd['Present_Weather_IceCrystals'].value_counts())
plt.bar(sampled_weather_pd['Present_Weather_IceCrystals'].value_counts().index, height = sampled_weather_pd['Present_Weather_IceCrystals'].value_counts()/sum(sampled_weather_pd['Present_Weather_IceCrystals'].value_counts()));
plt.xlabel('Ice Crystals (1 = Yes, 0 = No)');
ax.set_xticks([0, 1])
plt.ylabel('Proportion');
plt.title('Ice Crystals');

# COMMAND ----------

fig, ax = plt.subplots()
print(sampled_weather_pd['Present_Weather_Hail'].value_counts())
plt.bar(sampled_weather_pd['Present_Weather_Hail'].value_counts().index, height = sampled_weather_pd['Present_Weather_Hail'].value_counts()/sum(sampled_weather_pd['Present_Weather_Hail'].value_counts()));
plt.xlabel('Hail (1 = Yes, 0 = No)');
ax.set_xticks([0, 1])
plt.ylabel('Proportion');
plt.title('Hail');

# COMMAND ----------

fig, ax = plt.subplots()
print(sampled_weather_pd['Present_Weather_Mist'].value_counts())
plt.bar(sampled_weather_pd['Present_Weather_Mist'].value_counts().index, height = sampled_weather_pd['Present_Weather_Mist'].value_counts()/sum(sampled_weather_pd['Present_Weather_Mist'].value_counts()));
plt.xlabel('Mist (1 = Yes, 0 = No)');
ax.set_xticks([0, 1])
plt.ylabel('Proportion');
plt.title('Mist');

# COMMAND ----------

fig, ax = plt.subplots()
print(sampled_weather_pd['Present_Weather_Fog'].value_counts())
plt.bar(sampled_weather_pd['Present_Weather_Fog'].value_counts().index, height = sampled_weather_pd['Present_Weather_Fog'].value_counts()/sum(sampled_weather_pd['Present_Weather_Fog'].value_counts()));
plt.xlabel('Fog (1 = Yes, 0 = No)');
ax.set_xticks([0, 1])
plt.ylabel('Proportion');
plt.title('Fog');

# COMMAND ----------

fig, ax = plt.subplots()
print(sampled_weather_pd['Present_Weather_Smoke'].value_counts())
plt.bar(sampled_weather_pd['Present_Weather_Smoke'].value_counts().index, height = sampled_weather_pd['Present_Weather_Smoke'].value_counts()/sum(sampled_weather_pd['Present_Weather_Smoke'].value_counts()));
plt.xlabel('Smoke (1 = Yes, 0 = No)');
ax.set_xticks([0, 1])
plt.ylabel('Proportion');
plt.title('Smoke');

# COMMAND ----------

fig, ax = plt.subplots()
print(sampled_weather_pd['Present_Weather_Dust'].value_counts())
plt.bar(sampled_weather_pd['Present_Weather_Dust'].value_counts().index, height = sampled_weather_pd['Present_Weather_Dust'].value_counts()/sum(sampled_weather_pd['Present_Weather_Dust'].value_counts()));
plt.xlabel('Dust (1 = Yes, 0 = No)');
ax.set_xticks([0, 1])
plt.ylabel('Proportion');
plt.title('Dust');

# COMMAND ----------

fig, ax = plt.subplots()
print(sampled_weather_pd['Present_Weather_Haze'].value_counts())
plt.bar(sampled_weather_pd['Present_Weather_Haze'].value_counts().index, height = sampled_weather_pd['Present_Weather_Haze'].value_counts()/sum(sampled_weather_pd['Present_Weather_Haze'].value_counts()));
plt.xlabel('Haze (1 = Yes, 0 = No)');
ax.set_xticks([0, 1])
plt.ylabel('Proportion');
plt.title('Haze');

# COMMAND ----------

fig, ax = plt.subplots()
print(sampled_weather_pd['Present_Weather_Storm'].value_counts())
plt.bar(sampled_weather_pd['Present_Weather_Storm'].value_counts().index, height = sampled_weather_pd['Present_Weather_Storm'].value_counts()/sum(sampled_weather_pd['Present_Weather_Storm'].value_counts()));
plt.xlabel('Storm (1 = Yes, 0 = No)');
ax.set_xticks([0, 1])
plt.ylabel('Proportion');
plt.title('Storm');

# COMMAND ----------

print(sampled_weather_pd['HourlyPressureTendency_Increasing'].value_counts())
print(sampled_weather_pd['HourlyPressureTendency_Decreasing'].value_counts())
print(sampled_weather_pd['HourlyPressureTendency_Constant'].value_counts())

# COMMAND ----------

def count_pressure_tendency(var):
    sql = """
    SELECT """ + var + """, COUNT(""" + var + """) as cnt
    FROM weather_vw
    GROUP BY """ + var
    df = spark.sql(sql)
    display(df)
count_pressure_tendency('HourlyPressureTendency_Increasing')
count_pressure_tendency('HourlyPressureTendency_Decreasing')
count_pressure_tendency('HourlyPressureTendency_Constant')

# COMMAND ----------

fig, ax = plt.subplots()
print(sampled_weather_pd['HourlyPressureTendency_Increasing'].value_counts())
plt.bar(sampled_weather_pd['HourlyPressureTendency_Increasing'].value_counts().index, height = sampled_weather_pd['HourlyPressureTendency_Increasing'].value_counts()/sum(sampled_weather_pd['HourlyPressureTendency_Increasing'].value_counts()));
plt.xlabel('Pressure Increasing (1 = Yes, 0 = No)');
ax.set_xticks([0, 1])
plt.ylabel('Proportion');
plt.title('Pressure Increasing');

# COMMAND ----------

fig, ax = plt.subplots()
print(sampled_weather_pd['Calm_Winds'].value_counts())
plt.bar(sampled_weather_pd['Calm_Winds'].value_counts().index, height = sampled_weather_pd['Calm_Winds'].value_counts()/sum(sampled_weather_pd['Calm_Winds'].value_counts()));
plt.xlabel('Calm Winds (1 = Yes, 0 = No)');
ax.set_xticks([0, 1])
plt.ylabel('Proportion');
plt.title('Calm Winds');

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize = (10, 5))
fig.suptitle('Average Non-Zero Hourly Precipitation (inches)');
axes[0].set_title('Hourly Precipitation < 0.05');
sns.kdeplot(ax = axes[0], x = sampled_weather_pd[(sampled_weather_pd['Avg_Precip_Double'] > 0.00) &
                                  (sampled_weather_pd['Avg_Precip_Double'] < 0.05)]['Avg_Precip_Double']);
sns.kdeplot(ax = axes[1], x = sampled_weather_pd[(sampled_weather_pd['Avg_Precip_Double'] > 0.00) &
                                  (sampled_weather_pd['Avg_Precip_Double'] < 1)]['Avg_Precip_Double']);
axes[1].set_title('Hourly Precipitation < 1');

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize = (10, 5))
fig.suptitle('Average Wind Gust Speed (mph)');
axes[0].set_title('Average Wind Gust Speed < 50 mph');
sns.kdeplot(ax = axes[0], x = sampled_weather_pd[sampled_weather_pd['Avg_HourlyWindGustSpeed'] < 50]['Avg_HourlyWindGustSpeed']);
sns.kdeplot(ax = axes[1], x = sampled_weather_pd['Avg_HourlyWindGustSpeed']);
axes[1].set_title('Total Distribution');
# [(sampled_weather_pd['Avg_Precip_Double'] > 0.00) &
#                                   (sampled_weather_pd['Avg_Precip_Double'] < 0.05)]

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize = (15, 5))
fig.suptitle('Average Wind Speed (mph)');
axes[0].set_title('Average Wind Speed < 20 mph');
sns.kdeplot(ax = axes[0], x = sampled_weather_pd[sampled_weather_pd['Avg_HourlyWindSpeed'] < 20]['Avg_HourlyWindSpeed']);
sns.kdeplot(ax = axes[1], x = sampled_weather_pd[sampled_weather_pd['Avg_HourlyWindSpeed'] < 50]['Avg_HourlyWindSpeed']);
sns.kdeplot(ax = axes[2], x = sampled_weather_pd['Avg_HourlyWindSpeed']);
axes[1].set_title('Average Wind Speed < 50 mph');
axes[2].set_title('Total distribution shows erroneous data');
# [(sampled_weather_pd['Avg_Precip_Double'] > 0.00) &
#                                   (sampled_weather_pd['Avg_Precip_Double'] < 0.05)]

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize = (10, 5))
fig.suptitle('Average Hourly Pressure Change (in Hg)');
axes[0].set_title('Average Hourly Pressure Change < 0.1');
sns.kdeplot(ax = axes[0], x = sampled_weather_pd[abs(sampled_weather_pd['Avg_HourlyPressureChange']) < 0.1]['Avg_HourlyPressureChange']);
sns.kdeplot(ax = axes[1], x = sampled_weather_pd['Avg_HourlyPressureChange']);
axes[1].set_title('Total Distribution');
# [(sampled_weather_pd['Avg_Precip_Double'] > 0.00) &
#                                   (sampled_weather_pd['Avg_Precip_Double'] < 0.05)]

# COMMAND ----------

sky_conditions = ['Sky_Conditions_CLR', 'Sky_Conditions_FEW',
       'Sky_Conditions_SCT', 'Sky_Conditions_BKN', 'Sky_Conditions_OVC',
       'Sky_Conditions_VV']

for condition in sky_conditions:
    print(sampled_weather_pd[condition].value_counts())

# COMMAND ----------

sampled_weather_pd.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Joined Dataset

# COMMAND ----------

# Setup Blob store access
blob_container = "w261team11" # The name of your container created in https://portal.azure.com
storage_account = "w261sa" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261team11" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261team11key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"

FINAL_JOINED_DATA_ALL = f"{blob_url}/staged/final_joined_all"

# COMMAND ----------

# Load the 2015 Q1 for Weather
df = spark.read.parquet(FINAL_JOINED_DATA_ALL)
display(df)



# COMMAND ----------

sampled_joined = df.sample(fraction = 0.025)
sampled_joined_pd = sampled_joined.toPandas()
sampled_joined_pd

# COMMAND ----------

cols_to_keep_origin = ['DEP_DELAY', 'DEP_DEL15','origin_weather_Avg_Elevation', 'origin_weather_Avg_HourlyAltimeterSetting', 'origin_weather_Avg_HourlyDewPointTemperature', 'origin_weather_Avg_HourlyDryBulbTemperature', 'origin_weather_Avg_HourlyPressureChange', 'origin_weather_Avg_HourlyRelativeHumidity', 'origin_weather_Avg_HourlySeaLevelPressure', 'origin_weather_Avg_HourlyStationPressure', 'origin_weather_Avg_HourlyVisibility', 'origin_weather_Avg_HourlyWetBulbTemperature', 'origin_weather_Avg_HourlyWindDirection', 'origin_weather_Avg_HourlyWindGustSpeed', 'origin_weather_Avg_HourlyWindSpeed', 'origin_weather_Avg_Precip_Double']

cols_to_keep_dest = ['DEP_DELAY', 'DEP_DEL15','dest_weather_Avg_Elevation', 'dest_weather_Avg_HourlyAltimeterSetting', 'dest_weather_Avg_HourlyDewPointTemperature', 'dest_weather_Avg_HourlyDryBulbTemperature', 'dest_weather_Avg_HourlyPressureChange', 'dest_weather_Avg_HourlyRelativeHumidity', 'dest_weather_Avg_HourlySeaLevelPressure', 'dest_weather_Avg_HourlyStationPressure', 'dest_weather_Avg_HourlyVisibility', 'dest_weather_Avg_HourlyWetBulbTemperature', 'dest_weather_Avg_HourlyWindDirection', 'dest_weather_Avg_HourlyWindGustSpeed', 'dest_weather_Avg_HourlyWindSpeed', 'dest_weather_Avg_Precip_Double']

cols_to_keep_both = ['DEP_DELAY', 'DEP_DEL15','origin_weather_Avg_Elevation', 'origin_weather_Avg_HourlyAltimeterSetting', 'origin_weather_Avg_HourlyDewPointTemperature', 'origin_weather_Avg_HourlyDryBulbTemperature', 'origin_weather_Avg_HourlyPressureChange', 'origin_weather_Avg_HourlyRelativeHumidity', 'origin_weather_Avg_HourlySeaLevelPressure', 'origin_weather_Avg_HourlyStationPressure', 'origin_weather_Avg_HourlyVisibility', 'origin_weather_Avg_HourlyWetBulbTemperature', 'origin_weather_Avg_HourlyWindDirection', 'origin_weather_Avg_HourlyWindGustSpeed', 'origin_weather_Avg_HourlyWindSpeed', 'origin_weather_Avg_Precip_Double', 'dest_weather_Avg_Elevation', 'dest_weather_Avg_HourlyAltimeterSetting', 'dest_weather_Avg_HourlyDewPointTemperature', 'dest_weather_Avg_HourlyDryBulbTemperature', 'dest_weather_Avg_HourlyPressureChange', 'dest_weather_Avg_HourlyRelativeHumidity', 'dest_weather_Avg_HourlySeaLevelPressure', 'dest_weather_Avg_HourlyStationPressure', 'dest_weather_Avg_HourlyVisibility', 'dest_weather_Avg_HourlyWetBulbTemperature', 'dest_weather_Avg_HourlyWindDirection', 'dest_weather_Avg_HourlyWindGustSpeed', 'dest_weather_Avg_HourlyWindSpeed', 'dest_weather_Avg_Precip_Double']




df_joined_origin = sampled_joined_pd[cols_to_keep_origin]
df_joined_dest = sampled_joined_pd[cols_to_keep_dest]
df_joined_both = sampled_joined_pd[cols_to_keep_both]

# COMMAND ----------

corr_origin = df_joined_origin.corr()
corr_dest = df_joined_dest.corr()
corr_both = df_joined_both.corr()

# COMMAND ----------

cmap = sns.diverging_palette(230, 20, as_cmap=True)

# plot the heatmap
sns.heatmap(corr, 
        cmap=cmap,
        xticklabels=corr_both.columns,
        yticklabels=corr_both.columns)
plt.title('Weather vs Delay Correlation Matrix (Origin and Destination)');

# COMMAND ----------

cmap = sns.diverging_palette(230, 20, as_cmap=True)

# plot the heatmap
sns.heatmap(corr, 
        cmap=cmap,
        xticklabels=corr_both.columns,
        yticklabels=corr_both.columns)
plt.title('Weather vs Delay Correlation Matrix (Origin and Destination)');

# COMMAND ----------

sns.boxplot(x = sampled_joined_pd['DEP_DEL15'], 
            y = sampled_joined_pd['dest_weather_Avg_Elevation']);
plt.xlabel('15 min or Longer Delay');
plt.ylabel('Elevation');
plt.title('Elevation vs Delay');

# COMMAND ----------

sns.boxplot(x = sampled_joined_pd['DEP_DEL15'], 
            y = sampled_joined_pd['dest_weather_Avg_HourlyAltimeterSetting']);
plt.xlabel('15 min or Longer Delay');
plt.ylabel('Altimeter');
plt.title('Altimeter vs Delay');

# COMMAND ----------

sns.boxplot(x = sampled_joined_pd['DEP_DEL15'], 
            y = sampled_joined_pd['dest_weather_Avg_HourlyDewPointTemperature']);
plt.xlabel('15 min or Longer Delay');
plt.ylabel('Dew Point Temp');
plt.title('Dew Point Temp vs Delay');

# COMMAND ----------

sns.boxplot(x = sampled_joined_pd['DEP_DEL15'], 
            y = sampled_joined_pd['dest_weather_Avg_HourlyDryBulbTemperature']);
plt.xlabel('15 min or Longer Delay');
plt.ylabel('Bulb Temp');
plt.title('Bulb Temp vs Delay');

# COMMAND ----------

sns.boxplot(x = sampled_joined_pd['DEP_DEL15'], 
            y = sampled_joined_pd['dest_weather_Avg_HourlyPressureChange']);
plt.xlabel('15 min or Longer Delay');
plt.ylabel('Pressure Change');
plt.title('Pressure Change vs Delay');

# COMMAND ----------

sns.boxplot(x = sampled_joined_pd['DEP_DEL15'], 
            y = sampled_joined_pd['dest_weather_Avg_HourlyRelativeHumidity']);
plt.xlabel('15 min or Longer Delay');
plt.ylabel('Relative humidity');
plt.title('Relative Humdidity vs Delay');

# COMMAND ----------

sns.boxplot(x = sampled_joined_pd['DEP_DEL15'], 
            y = sampled_joined_pd['dest_weather_Avg_HourlyVisibility']);
plt.xlabel('15 min or Longer Delay');
plt.ylabel('Visibility');
plt.title('Visibility vs Delay');

# COMMAND ----------


