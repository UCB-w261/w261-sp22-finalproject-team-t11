# Databricks notebook source
from pyspark.sql.functions import col
# from pyspark.sql import SparkSession ? --> do we need this?
# do we need to build a Spark Context?
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pyspark
import datetime as dt
from pyspark.sql.functions import when

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

FINAL_JOINED_DATA_ALL = f"{blob_url}/staged/final_joined_all"

# COMMAND ----------

# Load the 2015 Q1 for Weather
df = spark.read.parquet(FINAL_JOINED_DATA_ALL)

display(df)


# COMMAND ----------

sampled_joined = df.sample(fraction = 0.025)
sampled_joined_pd = sampled_joined.toPandas()
#display(sampled_joined_pd)

# COMMAND ----------



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
sns.heatmap(corr_both, 
        cmap=cmap,
        xticklabels=corr_both.columns,
        yticklabels=corr_both.columns)
plt.title('Weather vs Delay Correlation Matrix (Origin and Destination)');

# COMMAND ----------

cmap = sns.diverging_palette(230, 20, as_cmap=True)

# plot the heatmap
sns.heatmap(corr_origin, 
        cmap=cmap,
        xticklabels=corr_origin.columns,
        yticklabels=corr_origin.columns)
plt.title('Origin Weather vs Delay Correlation Matrix');

# COMMAND ----------

cmap = sns.diverging_palette(230, 20, as_cmap=True)

# plot the heatmap
sns.heatmap(corr_dest, 
        cmap=cmap,
        xticklabels=corr_dest.columns,
        yticklabels=corr_dest.columns)
plt.title('Destination Weather vs Delay Correlation Matrix');

# COMMAND ----------

df_joined_origin.hist(figsize=(20,20), bins = 20)

# COMMAND ----------

df_joined_dest.hist(figsize=(20,20), bins = 20)

# COMMAND ----------

ax = sns.boxplot(x = sampled_joined_pd['origin_weather_Present_Weather_Hail'], 
            y = sampled_joined_pd['DEP_DELAY']);
plt.xlabel('Is Hail Present at Origin Airport');
plt.ylabel('Delay Time');
ax.set(ylim=(-60, 60));
plt.title('Delay vs Hail (Origin)');

# COMMAND ----------

ax = sns.boxplot(x = sampled_joined_pd['origin_weather_Present_Weather_Storm'], 
            y = sampled_joined_pd['DEP_DELAY']);
plt.xlabel('Is a Storm Present');
plt.ylabel('Delay Time');
ax.set(ylim=(-60, 60));
plt.title('Delay vs Storm');

# COMMAND ----------

ax = sns.boxplot(x = sampled_joined_pd['origin_weather_Present_Weather_Snow'], 
            y = sampled_joined_pd['DEP_DELAY']);
plt.xlabel('Is Snow Present');
plt.ylabel('Delay Time');
ax.set(ylim=(-60, 60));
plt.title('Delay vs Snow');

# COMMAND ----------


ax = sns.boxplot(x = sampled_joined_pd['origin_weather_Present_Weather_Rain'], 
            y = sampled_joined_pd['DEP_DELAY']);
plt.xlabel('Is Rain Present');
plt.ylabel('Delay Time');
ax.set(ylim=(-60, 60));
plt.title('Delay vs Rain');

# COMMAND ----------


