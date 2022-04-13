# Databricks notebook source
# MAGIC %md
# MAGIC # EDA on Flight Data

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
# MAGIC ## Load Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Flight Data

# COMMAND ----------

# Load 2015 Q1 for Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
df_airlines.createOrReplaceTempView("airlines_3m_vw")
display(df_airlines)

# COMMAND ----------

# Load all flights data. Big dataset. Be warned.
df_airlines_all = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*")
df_airlines_all.createOrReplaceTempView("airlines_vw")
display(df_airlines_all)

# COMMAND ----------

df_airlines_all.count()

# COMMAND ----------

def count_vals():
    sql = """
        SELECT COUNT(*)
        FROM weather_vw

    """
    
    result = spark.sql(sql)
    display(result)
    
count_vals()
#         WHERE year(DATEHOUR) >= 2020

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weather Data

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
# MAGIC ## Flights EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Inspection

# COMMAND ----------

# Inspect Time Period features
df_airlines_all.select("YEAR").distinct().show()

# COMMAND ----------

df_airlines_all.select("QUARTER").distinct().show()

# COMMAND ----------

df_airlines_all.select("MONTH").distinct().show()

# COMMAND ----------

#df_airlines_all.select("DAY_OF_MONTH").distinct().show()
df_airlines_all.select(countDistinct("DAY_OF_MONTH")).show()

# COMMAND ----------

# Inspect  OP_UNIQUE_CARRIER, OP_CARRIER_AIRLINE_ID and OP_CARRIER
# All 3 features contain the same information
# Recommend dropping the first two and keeping OP_CARRIER
df_airlines_all.select("OP_UNIQUE_CARRIER").distinct().show()

# COMMAND ----------

df_airlines_all.select(countDistinct("OP_UNIQUE_CARRIER")).show()

# COMMAND ----------

df_airlines_all.select(countDistinct("OP_CARRIER_AIRLINE_ID")).show()

# COMMAND ----------

df_airlines_all.select(countDistinct("OP_CARRIER")).show()

# COMMAND ----------

df_airlines_all.select("origin_airport_iso_country").distinct().show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Proportion of Delayed Flights

# COMMAND ----------

## Compare how many flights are delayed vs not delayed vs missing information
## In this cell, delays are based on departure time

# Aggregate the flights data
delayed_flights = df_airlines_all.groupBy("DEP_DEL15").count()

# Convert to Pandas DataFrame
delayed_flights_pd = delayed_flights.toPandas() # is there a better way to do this?

# Change display order by replacing nan's with 2, sort by the delay category
delayed_flights_pd_sorted = delayed_flights_pd.copy() # make a copy of the original data frame
delayed_flights_pd_sorted["DEP_DEL15"] = np.nan_to_num(delayed_flights_pd_sorted, nan = 2) # replace nan's with 2's
delayed_flights_pd_sorted = delayed_flights_pd_sorted.sort_values(by = "DEP_DEL15") # sort the result

# Plot the flight status
fig, ax = plt.subplots() # create the figure
fig1 = plt.bar(x = ["Not Delayed", "Delayed", "Missing Data"], height = delayed_flights_pd_sorted["count"]); # add the data with appropriate labels
plt.xlabel("Flight Status", fontsize = 12); # add x axis label
plt.ylabel("Number of Flights", fontsize = 12); # add y axis label
plt.title("Most flights are not delayed", fontsize = 14); # add title
# add percent distribution to figure
ax.bar_label(fig1, 
             labels = [('%.2f' % val) + '%' for val in delayed_flights_pd_sorted["count"] / sum(delayed_flights_pd_sorted["count"])*100], 
             label_type = 'edge', padding = 2);
# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights_pd_sorted["count"])*1.1);

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparing Departure and Arrival Delays

# COMMAND ----------

## Compare the extent to which flights are delayed for both departures and arrivals

# Define the appropriate column names in the dataframe
col_name_1 = 'DEP_DELAY_GROUP'
col_name_2 = 'ARR_DELAY_GROUP'

# Perform group by calculations for each column
delayed_departures = df_airlines.groupBy(col_name_1).count()
delayed_arrivals = df_airlines.groupBy(col_name_2).count()

# Convert to pandas and combine dataframes for plotting
delayed_departures_pd = delayed_departures.toPandas() # convert to pandas
delayed_arrivals_pd = delayed_arrivals.toPandas() # convert to pandas
# Copy the departure delay dataframe, rename the count column to 'Departure Delay', sort the values in ascending order
combined_delays_pd = delayed_departures_pd.copy().rename(columns={delayed_departures_pd.columns[1]: 'Departure Delay'}).sort_values(by = col_name_1)
combined_delays_pd['Arrival Delay'] = delayed_arrivals_pd.sort_values(by = col_name_2)['count'] # add arrival delay column
combined_delays_pd = combined_delays_pd.set_index(col_name_1) # set the index to equal the delay group value

# Create bar chart and set labels
combined_delays_pd.plot(kind = 'bar');
plt.xlabel("Delay Group");
plt.ylabel("Number of Flights");
plt.title("Comparison of Departure and Arrival Delays");

# COMMAND ----------

# MAGIC %md
# MAGIC ### Proportion of Delays by Destination Airport

# COMMAND ----------

display(df_airlines.groupBy('DEST').mean("DEP_DEL15"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Proportion of Delays by Time Period

# COMMAND ----------

## DELAYS BY TIME PERIOD
def delays_by_quarter(table):
    sql = """
        SELECT quarter, AVG(DEP_DEL15)
        FROM """ + table + """
        GROUP BY quarter
        ORDER BY quarter
    """
    
    result_df = spark.sql(sql)
    return result_df

def delays_by_month(table):
    sql = """
        SELECT month, AVG(DEP_DEL15)
        FROM """ + table + """
        GROUP BY month
        ORDER BY month
    """
    result_df = spark.sql(sql)
    return result_df

def delays_by_weekday(table):
    sql = """
        SELECT day_of_week, AVG(DEP_DEL15)
        FROM """ + table + """
        GROUP BY day_of_week
        ORDER BY day_of_week
    """
    result_df = spark.sql(sql)
    return result_df

display(delays_by_quarter('airlines_vw'))
display(delays_by_month('airlines_vw'))
display(delays_by_weekday('airlines_vw'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Proportion of Delays by Airline

# COMMAND ----------

# Delays by Airline
def delays_by_airline(table):
    sql = """
        SELECT op_unique_carrier, op_carrier, AVG(DEP_DEL15)
        FROM """ + table + """
        GROUP BY op_unique_carrier, op_carrier
        ORDER BY op_unique_carrier, op_carrier
    """
    
    result_df = spark.sql(sql)
    return result_df
display(delays_by_airline('airlines_vw'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Proportion of Delay by Flight Path

# COMMAND ----------

# Destination airport with highest average delay 
delays_by_destination = df_airlines.groupBy('ORIGIN','DEST').mean("DEP_DEL15").sort("avg(DEP_DEL15)", ascending=False).show()


# delays_by_destination_pd = delays_by_destination.toPandas()
# plt.bar(x = delays_by_destination_pd['DEST'], height = delays_by_destination_pd['avg(DEP_DEL15)'])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Proportion of Delay by Destination State

# COMMAND ----------

delays_by_destination_state = df_airlines.groupBy('DEST_STATE_ABR').mean("DEP_DEL15")
delays_by_destination_state_pd = delays_by_destination_state.toPandas().sort_values(by = 'DEST_STATE_ABR')
fig, ax = plt.subplots(figsize = (20, 5));
plt.bar(x = delays_by_destination_state_pd['DEST_STATE_ABR'], height = delays_by_destination_state_pd['avg(DEP_DEL15)']);
plt.xlabel('State Abbreviation');
plt.title('Proporition of Flights that are Delayed by State');

# COMMAND ----------

# Arrival airport with highest average delay ***use as a feature in model***
delays_by_arrival = df_airlines.groupBy('ORIGIN').mean("DEP_DEL15").sort("avg(DEP_DEL15)", ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Limits on Departure Times

# COMMAND ----------

# Looking at departure time: earliest departure time is 5am, appears to be an integer
df_airlines.agg({'CRS_DEP_TIME':'min'}).show()

# COMMAND ----------

# Looking at departure time: latest departure time is 11:10pm, appears to be an integer
df_airlines.agg({'CRS_DEP_TIME':'max'}).show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Proportion of Delay by Origin State

# COMMAND ----------

delays_by_destination_state = df_airlines.groupBy('ORIGIN_STATE_ABR').mean("DEP_DEL15")
delays_by_destination_state_pd = delays_by_destination_state.toPandas().sort_values(by = 'ORIGIN_STATE_ABR')
fig, ax = plt.subplots();
plt.bar(x = delays_by_destination_state_pd['ORIGIN_STATE_ABR'], height = delays_by_destination_state_pd['avg(DEP_DEL15)']);
plt.xlabel('State Abbreviation');
plt.title('Proporition of Flights that are Delayed by State');

# COMMAND ----------

# MAGIC %md
# MAGIC ## Joined EDA

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


