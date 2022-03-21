# Databricks notebook source
# MAGIC %md
# MAGIC # Weather Cleaning

# COMMAND ----------

len(dbutils.fs.ls('/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2015'))
# dbutils"/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/temp/2015/"

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
from pyspark.sql.functions import when

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather Data

# COMMAND ----------

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")
df_weather.createOrReplaceTempView("weather_vw")
display(df_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ram's List of Stations

# COMMAND ----------

closest_stations = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/jconde@berkeley.edu/closest_ws_stations.csv")
display(closest_stations)
closest_stations.createOrReplaceTempView("closest_stations")

# COMMAND ----------

closest_stations.show()

# COMMAND ----------

# Filter original weather data to only show stations in the closest station list, and specifying columns
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

# WIND COLUMN
# split the column and drop the original
# as appropriate, convert number values to floats and apply scaling factors
wind_split = pyspark.sql.functions.split(weather_filtered['WND'], ',')
weather_filtered_new = weather_filtered.withColumn('WND_DirectionAngle', wind_split.getItem(0).cast("float"))
weather_filtered_new = weather_filtered_new.withColumn('WND_DirectionQuality', wind_split.getItem(1).cast("float"))
weather_filtered_new = weather_filtered_new.withColumn('WND_Type', wind_split.getItem(2))
weather_filtered_new = weather_filtered_new.withColumn('WND_Speed', wind_split.getItem(3).cast("float")/10)
weather_filtered_new = weather_filtered_new.withColumn('WND_SpeedQuality', wind_split.getItem(4))
weather_filtered_new = weather_filtered_new.drop('WND')

# CEILING HEIGHT
# split the column and drop the original
# as appropriate, convert number values to floats and apply scaling factors
cig_split = pyspark.sql.functions.split(weather_filtered['CIG'], ',')
weather_filtered_new = weather_filtered_new.withColumn('CIG_CeilingHeightDim', cig_split.getItem(0).cast("float"))
weather_filtered_new = weather_filtered_new.withColumn('CIG_CeilingQuality', cig_split.getItem(1))
weather_filtered_new = weather_filtered_new.withColumn('CIG_CeilingDetermination', cig_split.getItem(2))
weather_filtered_new = weather_filtered_new.withColumn('CIG_CeilingAndVisibility', cig_split.getItem(3))
weather_filtered_new = weather_filtered_new.drop('CIG')

# HORIZONTAL VISIBILITY
# split the column and drop the original
# as appropriate, convert number values to floats and apply scaling factors
vis_split = pyspark.sql.functions.split(weather_filtered['VIS'], ',')
weather_filtered_new = weather_filtered_new.withColumn('VIS_Horizontal', vis_split.getItem(0).cast("float"))
weather_filtered_new = weather_filtered_new.withColumn('VIS_DistanceQuality', vis_split.getItem(1))
weather_filtered_new = weather_filtered_new.withColumn('VIS_Variability', vis_split.getItem(2))
weather_filtered_new = weather_filtered_new.withColumn('VIS_QualityOfVariability', vis_split.getItem(3))
weather_filtered_new = weather_filtered_new.drop('VIS')

# TEMPERATURE
# split the column and drop the original
# as appropriate, convert number values to floats and apply scaling factors
tmp_split = pyspark.sql.functions.split(weather_filtered['TMP'], ',')
weather_filtered_new = weather_filtered_new.withColumn('TMP_tmp', tmp_split.getItem(0).cast("float")/10)
weather_filtered_new = weather_filtered_new.withColumn('TMP_Quality', tmp_split.getItem(1))
weather_filtered_new = weather_filtered_new.drop('TMP')

# DEW POINT
# split the column and drop the original
# as appropriate, convert number values to floats and apply scaling factors
dew_split = pyspark.sql.functions.split(weather_filtered['DEW'], ',')
weather_filtered_new = weather_filtered_new.withColumn('DEW_dew', dew_split.getItem(0).cast("float")/10)
weather_filtered_new = weather_filtered_new.withColumn('DEW_Quality', dew_split.getItem(1))
weather_filtered_new = weather_filtered_new.drop('DEW')

# AIR PRESSURE
# split the column and drop the original
# as appropriate, convert number values to floats and apply scaling factors
slp_split = pyspark.sql.functions.split(weather_filtered['SLP'], ',')
weather_filtered_new = weather_filtered_new.withColumn('SLP_slp', slp_split.getItem(0).cast("float")/10)
weather_filtered_new = weather_filtered_new.withColumn('SLP_Quality', slp_split.getItem(1))
weather_filtered_new = weather_filtered_new.drop('SLP')

# PRECIPITATION
# get the precipitation in mm and the hours of measurement
# convert to mm/hr value (?)
# drop the original columns
# Precipitation actually has AA1 through AA4 but I think most of the relevant data is in AA1 and 
# the other columns are just supplementary to this
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
# There are actually AT1 through AT8 but past AT4 things seemed to be pretty empty, not sure how these are ordered
# We can switch this to pull through AT8 if needed
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

display(weather_filtered_new)
weather_filtered_new.createOrReplaceTempView("weather_filtered_vw")

# COMMAND ----------



# COMMAND ----------

# Since you can't overwrite the old file, delete it first
try:
    %fs rm -r "/FileStore/shared_uploads/jconde@berkeley.edu/weather_filtered.csv"
except:
    pass

weather_filtered_new.write.csv("dbfs:/FileStore/shared_uploads/jconde@berkeley.edu/weather_filtered.csv", header='true')
