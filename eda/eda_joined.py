# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Notebook Setup

# COMMAND ----------

from pyspark.sql.functions import col, isnan, when, count, concat, lit, sum
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pyspark
import datetime as dt
import pandas as pd

# Setup Blob store access
# Original given sources:
WEATHER_RAW_LOC = "/mnt/mids-w261/datasets_final_project/weather_data/*"

# New data sources:
FLIGHT_COVID_RAW_LOC = f"{blob_url}/raw/flights_covid"
# WEATHER_STATIONS_RAW_LOC = f"{blob_url}/raw/stations"
WEATHER_RAW_LOC = f"{blob_url}/raw/weather"
AIRLINE_CODES_RAW_LOC = "dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/new_data/airline_codes/iata_airline_codes.csv"

# Location of staged data.
# AIRPORT_WEATHER_LOC = f"{blob_url}/raw/airport_weather"
AIRPORTS_MASTER_LOC = f"{blob_url}/staged/airports"
AIRPORTS_WS_LOC = f"{blob_url}/staged/airports_weatherstations"
WEATHER_LOC = f"{blob_url}/staged/weather"
CLEAN_WEATHER_LOC = f'{WEATHER_LOC}/clean_weather_data.parquet'

# Location of final joined data (2015-2018)
# FINAL_JOINED_DATA_ALL = f"{blob_url}/staged/final_joined_all"
FINAL_JOINED_DATA_TRAINING = f"{blob_url}/staged/final_joined_training"
# FINAL_JOINED_DATA_VALIDATION
# 2019 flights data used for testing model.
FINAL_JOINED_DATA_TEST = f"{blob_url}/staged/final_joined_testing"
# 2020-2021 covid era flights for additional exploration/testing.
FINAL_JOINED_DATA_20_21 = f"{blob_url}/staged/final_joined_2020_2021"

SHAPES_BASE_FOLDER = "/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/shapes"

# COMMAND ----------

# Load the joined data (training)
df = spark.read.parquet(FINAL_JOINED_DATA_TRAINING)
airline_codes = spark.read.format("csv").option("header", "true").load(AIRLINE_CODES_RAW_LOC)
airline_codes = airline_codes.withColumnRenamed('IATA Code', 'OP_UNIQUE_CARRIER')
df = df.join(airline_codes, on='OP_UNIQUE_CARRIER', how='inner')
display(df)


# COMMAND ----------

# Load the joined data (2019 - test set)
df_2019 = spark.read.parquet(FINAL_JOINED_DATA_TEST)
df_2019 = df_2019.join(airline_codes, on='OP_UNIQUE_CARRIER', how='inner')
display(df_2019)

# COMMAND ----------

# Load the joined data (2019 - test set)
df_covid = spark.read.parquet(FINAL_JOINED_DATA_20_21)
df_covid = df_covid.join(airline_codes, on='OP_UNIQUE_CARRIER', how='inner')
display(df_covid)

# COMMAND ----------

df_2020 = df_covid.filter(df_covid['YEAR']==2020).cache()
df_2021 = df_covid.filter(df_covid['YEAR']==2021).cache()

# COMMAND ----------

# check for outliers in columns used in the model
display(df.describe(['origin_weather_Present_Weather_Hail', 
                'origin_weather_Present_Weather_Storm', 
                'dest_weather_Present_Weather_Hail', 
                'dest_weather_Present_Weather_Storm',
                'QUARTER', 
                'MONTH', 
                'DAY_OF_WEEK', 
                'DISTANCE_GROUP', 
                'origin_airport_type', 
                'dest_airport_type']))

# COMMAND ----------

display(df.describe(['origin_weather_Avg_HourlyWindSpeed']))

# COMMAND ----------

df2 = df.sort('origin_weather_Avg_HourlyWindSpeed').head(200)

# COMMAND ----------

pd.DataFrame(df2, columns=df.schema)

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary

# COMMAND ----------

# count rows and columns in df
print(f'Training Dataset: {df.count()}, {len(df.columns)}')
print(f'Test Dataset:     {df_2019.count()}, {len(df_2019.columns)}')
print(f'COVID-19 Dataset: {df_covid.count()}, {len(df_covid.columns)}')
print(f'2020 Data):       {df_2020.count()}, {len(df_2020.columns)}')
print(f'2021 Data):       {df_2021.count()}, {len(df_2021.columns)}')


# COMMAND ----------

# inspect schema
df.printSchema()

# COMMAND ----------

# inspect column datatypes
# DoubleType: Represents 8-byte double-precision floating point numbers.
# https://spark.apache.org/docs/latest/sql-ref-datatypes.html
def count_column_types(spark_df):
    """Count number of columns per type"""
    return pd.DataFrame(spark_df.dtypes).groupby(1, as_index=False)[0].agg({'count':'count', 'names': lambda x: " | ".join(set(x))}).rename(columns={1:"type"})
column_types = count_column_types(df)
column_types

# COMMAND ----------

display(df.groupBy("YEAR").count().withColumnRenamed('count', 'Number of flights'))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Delay (Target Variable) Analysis

# COMMAND ----------

## Compare how many flights are delayed vs not delayed vs missing information

## In this cell, delays are based on departure time
import matplotlib.ticker as ticker

# Aggregate the flights data
delayed_flights = df.groupBy("DEP_DEL15").count()

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
             labels = [('%.2f' % val) + '%' for val in delayed_flights_pd_sorted["count"] / delayed_flights_pd_sorted["count"].sum()*100], 
             label_type = 'edge', padding = 2);
# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights_pd_sorted["count"])*1.1);
# fix y-tick values
ax.get_xticks()
ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights_pd_sorted["count"]))
ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights_pd_sorted["count"]])
fig.show()

# COMMAND ----------

## Compare how many flights are delayed vs not delayed vs missing information

## In this cell, delays are based on departure time
import matplotlib.ticker as ticker

# Aggregate the flights data
delayed_train = df.fillna(2, subset='DEP_DEL15').groupby('DEP_DEL15').count().sort('DEP_DEL15').toPandas()
delayed_2019 = df_2019.fillna(2, subset='DEP_DEL15').groupby('DEP_DEL15').count().sort('DEP_DEL15').toPandas()
delayed_2020 = df_covid.filter(df_covid['YEAR']==2020).fillna(2, subset='DEP_DEL15').groupby('DEP_DEL15').count().sort('DEP_DEL15').toPandas()
delayed_2021 = df_covid.filter(df_covid['YEAR']==2021).fillna(2, subset='DEP_DEL15').groupby('DEP_DEL15').count().sort('DEP_DEL15').toPandas()

# Plot the flight status
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(24,8))

fig1 = ax1.bar(x = ["Not Delayed", "Delayed", "Missing Data"], height = delayed_train["count"] / delayed_train["count"].sum()*100)
ax1.set_xlabel("Flight Status", fontsize = 12)
ax1.set_ylabel("Number of Flights", fontsize = 12)
ax1.set_title("Training data: 2015-2018", fontsize = 14)

ax1.bar_label(fig1, labels = [('%.2f' % val) + '%' for val in delayed_train["count"] / delayed_train["count"].sum()*100], 
             label_type = 'edge', padding = 2);
# ax1.set_ylim(top = max(delayed_train["count"])*1.1);
ax1.get_xticks()
# ax1.yaxis.set_major_locator(ticker.FixedLocator(delayed_train["count"]))
# ax1.set_yticklabels(['{:.0f}'.format(x) for x in delayed_train["count"]])

# Test data - 2019 only
fig2 = ax2.bar(x = ["Not Delayed", "Delayed", "Missing Data"], height = delayed_2019["count"] / delayed_2019["count"].sum()*100)
ax2.set_xlabel("Flight Status", fontsize = 12)
# ax2.set_ylabel("Number of Flights", fontsize = 12)
ax2.set_title("2019", fontsize = 14)

ax2.bar_label(fig2, labels = [('%.2f' % val) + '%' for val in delayed_2019["count"] / delayed_2019["count"].sum()*100], 
             label_type = 'edge', padding = 2);
# ax2.set_ylim(top = max(delayed_2019["count"])*1.1);
ax2.get_xticks()
# ax2.yaxis.set_major_locator(ticker.FixedLocator(delayed_2019["count"]))
# ax2.set_yticklabels(['{:.0f}'.format(x) for x in delayed_2019["count"]])

# Test data - 2020
fig3 = ax3.bar(x = ["Not Delayed", "Delayed", "Missing Data"], height = delayed_2020["count"] / delayed_2020["count"].sum()*100)
ax3.set_xlabel("Flight Status", fontsize = 12)
# ax3.set_ylabel("Number of Flights", fontsize = 12)
ax3.set_title("2020", fontsize = 14)

ax3.bar_label(fig3, labels = [('%.2f' % val) + '%' for val in delayed_2020["count"] / delayed_2020["count"].sum()*100], 
             label_type = 'edge', padding = 2);
# ax3.set_ylim(top = max(delayed_2020["count"])*1.1);
ax3.get_xticks()
# ax3.yaxis.set_major_locator(ticker.FixedLocator(delayed_2020["count"]))
# ax3.set_yticklabels(['{:.0f}'.format(x) for x in delayed_2020["count"]])

# Test data - 2021
fig4 = ax4.bar(x = ["Not Delayed", "Delayed", "Missing Data"], height = delayed_2021["count"] / delayed_2021["count"].sum()*100)
ax4.set_xlabel("Flight Status", fontsize = 12)
# ax4.set_ylabel("Number of Flights", fontsize = 12)
ax4.set_title("2021", fontsize = 14)

ax4.bar_label(fig4, labels = [('%.2f' % val) + '%' for val in delayed_2021["count"] / delayed_2021["count"].sum()*100], 
             label_type = 'edge', padding = 2);
# ax4.set_ylim(top = max(delayed_2021["count"])*1.1);
ax4.get_xticks()
# ax4.yaxis.set_major_locator(ticker.FixedLocator(delayed_2021["count"]))
# ax4.set_yticklabels(['{:.0f}'.format(x) for x in delayed_2021["count"]])

fig.show()

# COMMAND ----------

import matplotlib.ticker as ticker
# Aggregate the flights data
delayed_train = df.fillna(2, subset='DEP_DEL15').groupby('DEP_DEL15').count().sort('DEP_DEL15').toPandas()
delayed_2019 = df_2019.fillna(2, subset='DEP_DEL15').groupby('DEP_DEL15').count().sort('DEP_DEL15').toPandas()
delayed_2020 = df_covid.filter(df_covid['YEAR']==2020).fillna(2, subset='DEP_DEL15').groupby('DEP_DEL15').count().sort('DEP_DEL15').toPandas()
delayed_2021 = df_covid.filter(df_covid['YEAR']==2021).fillna(2, subset='DEP_DEL15').groupby('DEP_DEL15').count().sort('DEP_DEL15').toPandas()

labels = ["Not Delayed", "Delayed", "Missing Data"]
x = np.arange(len(labels))
w = 0.22

fig, ax = plt.subplots(figsize=(10,5))

fig1 = ax.bar(x = x - 3*w/2, height = delayed_train["count"] / delayed_train["count"].sum()*100, width=w, label='2015-18')
fig2 = ax.bar(x = x - w/2, height = delayed_2019["count"] / delayed_2019["count"].sum()*100, width=w, label='2019')
fig3 = ax.bar(x = x + w/2, height = delayed_2020["count"] / delayed_2020["count"].sum()*100, width=w, label='2020')
fig4 = ax.bar(x = x + 3*w/2, height = delayed_2021["count"] / delayed_2021["count"].sum()*100, width=w, label='2021')

ax.set_ylabel("Percentage of Flights", fontsize = 12)
ax.set_title("Percentage of flight delays by year", fontsize = 16)
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(fig1, labels = [('%.2f' % val) + '%' for val in delayed_train["count"] / delayed_train["count"].sum()*100], 
             label_type = 'edge', padding = 2, fontsize=12)
ax.bar_label(fig2, labels = [('%.2f' % val) + '%' for val in delayed_2019["count"] / delayed_2019["count"].sum()*100], 
             label_type = 'edge', padding = 2, fontsize=12)
ax.bar_label(fig3, labels = [('%.2f' % val) + '%' for val in delayed_2020["count"] / delayed_2020["count"].sum()*100], 
             label_type = 'edge', padding = 2, fontsize=12)
ax.bar_label(fig4, labels = [('%.2f' % val) + '%' for val in delayed_2021["count"] / delayed_2021["count"].sum()*100], 
             label_type = 'edge', padding = 2, fontsize=12)

fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cancelled Flights

# COMMAND ----------

import matplotlib.ticker as ticker

# Aggregate the flights data
delayed_train = df.fillna(0, subset='CANCELLED').groupby('CANCELLED').count().sort('CANCELLED').toPandas()
delayed_2019 = df_2019.fillna(0, subset='CANCELLED').groupby('CANCELLED').count().sort('CANCELLED').toPandas()
delayed_2020 = df_covid.filter(df_covid['YEAR']==2020).fillna(2, subset='CANCELLED').groupby('CANCELLED').count().sort('CANCELLED').toPandas()
delayed_2021 = df_covid.filter(df_covid['YEAR']==2021).fillna(2, subset='CANCELLED').groupby('CANCELLED').count().sort('CANCELLED').toPandas()

# Plot the flight status
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(24,8))

fig1 = ax1.bar(x = ["Regularly operated", "Canceled"], height = delayed_train["count"] / delayed_train["count"].sum()*100)
ax1.set_xlabel("Flight Status", fontsize = 12)
ax1.set_ylabel("Number of Flights", fontsize = 12)
ax1.set_title("Training data: 2015-2018)", fontsize = 14)

ax1.bar_label(fig1, labels = [('%.2f' % val) + '%' for val in delayed_train["count"] / delayed_train["count"].sum()*100], 
             label_type = 'edge', padding = 2);
# ax1.set_ylim(top = max(delayed_train["count"])*1.1);
ax1.get_xticks()
# ax1.yaxis.set_major_locator(ticker.FixedLocator(delayed_train["count"]))
# ax1.set_yticklabels(['{:.0f}'.format(x) for x in delayed_train["count"]])

# Test data - 2019 only
fig2 = ax2.bar(x = ["Regularly operated", "Canceled"], height = delayed_2019["count"] / delayed_2019["count"].sum()*100)
ax2.set_xlabel("Flight Status", fontsize = 12)
# ax2.set_ylabel("Number of Flights", fontsize = 12)
ax2.set_title("2019", fontsize = 14)

ax2.bar_label(fig2, labels = [('%.2f' % val) + '%' for val in delayed_2019["count"] / delayed_2019["count"].sum()*100], 
             label_type = 'edge', padding = 2);
# ax2.set_ylim(top = max(delayed_2019["count"])*1.1);
ax2.get_xticks()
# ax2.yaxis.set_major_locator(ticker.FixedLocator(delayed_2019["count"]))
# ax2.set_yticklabels(['{:.0f}'.format(x) for x in delayed_2019["count"]])

# Test data - 2020
fig3 = ax3.bar(x = ["Regularly operated", "Canceled"], height = delayed_2020["count"] / delayed_2020["count"].sum()*100)
ax3.set_xlabel("Flight Status", fontsize = 12)
# ax3.set_ylabel("Number of Flights", fontsize = 12)
ax3.set_title("2020", fontsize = 14)

ax3.bar_label(fig3, labels = [('%.2f' % val) + '%' for val in delayed_2020["count"] / delayed_2020["count"].sum()*100], 
             label_type = 'edge', padding = 2);
# ax3.set_ylim(top = max(delayed_2020["count"])*1.1);
ax3.get_xticks()
# ax3.yaxis.set_major_locator(ticker.FixedLocator(delayed_2020["count"]))
# ax3.set_yticklabels(['{:.0f}'.format(x) for x in delayed_2020["count"]])

# Test data - 2021
fig4 = ax4.bar(x = ["Regularly operated", "Canceled"], height = delayed_2021["count"] / delayed_2021["count"].sum()*100)
ax4.set_xlabel("Flight Status", fontsize = 12)
# ax4.set_ylabel("Number of Flights", fontsize = 12)
ax4.set_title("2021", fontsize = 14)

ax4.bar_label(fig4, labels = [('%.2f' % val) + '%' for val in delayed_2021["count"] / delayed_2021["count"].sum()*100], 
             label_type = 'edge', padding = 2);
# ax4.set_ylim(top = max(delayed_2021["count"])*1.1);
ax4.get_xticks()
# ax4.yaxis.set_major_locator(ticker.FixedLocator(delayed_2021["count"]))
# ax4.set_yticklabels(['{:.0f}'.format(x) for x in delayed_2021["count"]])

fig.show()

# COMMAND ----------

# Aggregate the flights data
delayed_train = df.fillna(0, subset='CANCELLED').groupby('CANCELLED').count().sort('CANCELLED').toPandas()
delayed_2019 = df_2019.fillna(0, subset='CANCELLED').groupby('CANCELLED').count().sort('CANCELLED').toPandas()
delayed_2020 = df_covid.filter(df_covid['YEAR']==2020).fillna(0, subset='CANCELLED').groupby('CANCELLED').count().sort('CANCELLED').toPandas()
delayed_2021 = df_covid.filter(df_covid['YEAR']==2021).fillna(0, subset='CANCELLED').groupby('CANCELLED').count().sort('CANCELLED').toPandas()

labels = ["Regularly operated", "Cancelled"]
x = np.arange(len(labels))
w = 0.22

fig, ax = plt.subplots(figsize=(8,5))

fig1 = ax.bar(x = x - 3*w/2, height = delayed_train["count"] / delayed_train["count"].sum()*100, width=w, label='2015-18')
fig2 = ax.bar(x = x - w/2, height = delayed_2019["count"] / delayed_2019["count"].sum()*100, width=w, label='2019')
fig3 = ax.bar(x = x + w/2, height = delayed_2020["count"] / delayed_2020["count"].sum()*100, width=w, label='2020')
fig4 = ax.bar(x = x + 3*w/2, height = delayed_2021["count"] / delayed_2021["count"].sum()*100, width=w, label='2021')

ax.set_ylabel("Percentage of Flights", fontsize = 12)
ax.set_title("Percentage of flight cancellations by year", fontsize = 16)
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(fig1, labels = [('%.2f' % val) + '%' for val in delayed_train["count"] / delayed_train["count"].sum()*100], 
             label_type = 'edge', padding = 2)
ax.bar_label(fig2, labels = [('%.2f' % val) + '%' for val in delayed_2019["count"] / delayed_2019["count"].sum()*100], 
             label_type = 'edge', padding = 2)
ax.bar_label(fig3, labels = [('%.2f' % val) + '%' for val in delayed_2020["count"] / delayed_2020["count"].sum()*100], 
             label_type = 'edge', padding = 2);
ax.bar_label(fig4, labels = [('%.2f' % val) + '%' for val in delayed_2021["count"] / delayed_2021["count"].sum()*100], 
             label_type = 'edge', padding = 2);

# ax.get_xticks()

fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Cancellations by month in 2020

# COMMAND ----------

cancellations_2020 = df_covid.filter(df_covid['YEAR']==2020).fillna(0, subset='CANCELLED').groupby('CANCELLED', 'MONTH').count().sort('MONTH')
display(cancellations_2020)

# COMMAND ----------

# display(cancellations_2020.groupBy('MONTH').sum('count').sort('MONTH'))

cancellations_2020 = cancellations_2020.alias('df_1').join(cancellations_2020.alias('df_2'), \
                                                          (col('df_1.MONTH') == col('df_2.MONTH')) & (col('df_1.CANCELLED')==0) & (col('df_2.CANCELLED')==1)) \
            .select(col('df_1.MONTH'), col('df_1.count').alias('operated'), col('df_2.count').alias('cancelled'))
cancellations_2020 = cancellations_2020.withColumn('total', cancellations_2020.operated + cancellations_2020.cancelled)
cancellations_2020 = cancellations_2020.withColumn('percentage_cancelled', cancellations_2020.cancelled/cancellations_2020.total)
display(cancellations_2020.sort('percentage_cancelled', ascending=False))

# COMMAND ----------

cancellations = cancellations_2020.sort('MONTH').toPandas()

labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
x = np.arange(len(labels))
w = 0.22

fig, ax = plt.subplots(figsize=(8,5))

fig1 = plt.bar(x = x, height = cancellations["percentage_cancelled"])

ax.set_ylabel("% of cancelled flights", fontsize = 12)
ax.set_title("Frequency of flight cancellations in 2020", fontsize = 16)
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(fig1, 
             labels = [('%.2f' % val) + '%' for val in cancellations["percentage_cancelled"]*100], 
             label_type = 'edge', padding = 2);

# ax.set_ylim(top = max(cancellations["percentage_cancelled"])*1.1);
fig.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delay by hour of day

# COMMAND ----------

## Plot average number of flights delayed by hour in day

def bucket(x):
    #returns the hour of departure
    x=int(x)
    x=int(x/100)
    return x

bucketudf = udf(lambda x: bucket(x))

df_delaybyhour = df.withColumn('HOUR', bucketudf(col('CRS_DEP_TIME'))).groupby('HOUR').mean("DEP_DEL15").sort("HOUR", ascending=False).toPandas()

# change hour to float
df_delaybyhour['HOUR'] = df_delaybyhour['HOUR'].astype(float)

# Plotting average # flights delayed by hour in day
df_delaybyhour.plot.scatter(x='HOUR', y='avg(DEP_DEL15)');
plt.ylabel('Proportion of Delayed Flights');
plt.title('Hour of day shows some positive correlation \n with flight delay frequency');

# COMMAND ----------

# MAGIC %md
# MAGIC # Nulls

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col
def null_check():
    display(df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]))

null_check()

# COMMAND ----------

# count nulls and nans in each column
import pyspark.sql.functions as F
def count_missings(spark_df,sort=True):
    """
    Counts number of nulls and nans in each column
    """
    _df = spark_df.select([F.count(F.when(F.isnan(c) | F.isnull(c), c)).alias(c) for (c,c_type) in spark_df.dtypes if c_type not in ('timestamp', 'string', 'date')]).toPandas()

    if len(_df) == 0:
        print("There are not any missing values!")
        return None

    if sort:
        return _df.rename(index={0: 'count'}).T.set_index(_df.columns).sort_values("count",ascending=False)

    return _df

missing_df = count_missings(df)

#nulls = count_missings(df)

#nulls.to_csv("nulls.csv", sep='\t')


# COMMAND ----------

# identify columns with over 20 million null/nan values
missing_df[missing_df['count']>20000000].index

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get small sample

# COMMAND ----------

sampled_joined = df.sample(fraction = 0.025)
sampled_joined_pd = sampled_joined.toPandas()
#display(sampled_joined_pd)

# COMMAND ----------

df_joined_origin = sampled_joined_pd[cols_to_keep_origin]
df_joined_dest = sampled_joined_pd[cols_to_keep_dest]
df_joined_both = sampled_joined_pd[cols_to_keep_both]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Corelations of the small sample

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

# MAGIC %md
# MAGIC ### Histograms of the small sample

# COMMAND ----------

df_joined_origin.hist(figsize=(20,20), bins = 20)

# COMMAND ----------

df_joined_dest.hist(figsize=(20,20), bins = 20)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Individual corelations

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

# MAGIC %sql
# MAGIC 
# MAGIC SELECT FL_DATE, TAIL_NUM, OP_UNIQUE_CARRIER, OP_CARRIER_FL_NUM, ORIGIN, ORIGIN_CITY_NAME, DEST, DEST_CITY_NAME, DEP_DEL15
# MAGIC FROM vw_joined_2015_2019
# MAGIC WHERE TAIL_NUM in ('ALL', 'PLANET', 'D942DN', 'N0EGMQ', 'SS25')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Find Outliers on Full Dataset

# COMMAND ----------

df_origin = df[cols_to_keep_origin]

# COMMAND ----------

df_origin = df_origin.toPandas()

# COMMAND ----------

df_origin.describe()

# COMMAND ----------

outliers = df_origin[df_origin > df_origin.mean() + 3 * df_origin.std()]

# COMMAND ----------

df_origin.hist(figsize=(20,20), bins = 20)

# COMMAND ----------

df_dest = df[cols_to_keep_dest]

# COMMAND ----------

df_dest = df_dest.toPandas()

# COMMAND ----------

df_dest.describe()

# COMMAND ----------

df_dest.hist(figsize=(20,20), bins = 20)

# COMMAND ----------

from scipy import stats


# COMMAND ----------

df_dest[(np.abs(stats.zscore(df_dest)) < 3).all(axis=1)]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Additional EDAs

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Flights to and from nowhere

# COMMAND ----------

print ("Total number of ORIGIN airports:", df.select("ORIGIN").distinct().count())
print ("Total number of DESTINATION airports:", df.select("DEST").distinct().count())


# COMMAND ----------

origin_airports = df.select("ORIGIN").distinct().sort('ORIGIN').withColumnRenamed('ORIGIN', 'AIRPORT').toPandas()
destination_airports = df.select("DEST").distinct().sort('DEST').withColumnRenamed('DEST', 'AIRPORT').toPandas()


# COMMAND ----------

origin_airports['AIRPORT']

# COMMAND ----------

[item for item in list(origin_airports['AIRPORT']) if item not in list(destination_airports['AIRPORT'])]

# COMMAND ----------

[item for item in list(destination_airports['AIRPORT']) if item not in list(origin_airports['AIRPORT'])]

# COMMAND ----------

display(df.where((df['ORIGIN']=='EFD') | (df['ORIGIN']=='ENV') | (df['DEST']=='FNL')))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Delayed flights by delay group

# COMMAND ----------

display(df.describe('DEP_DELAY_GROUP'))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.groupBy("DEP_DELAY_GROUP").count().drop(df['DEP_DELAY_GROUP']<=0).dropna('DEP_DELAY_GROUP').sort_values(by="DEP_DELAY_GROUP"))

# COMMAND ----------

display(df.groupby('DEP_DELAY_GROUP').count().filter(df['DEP_DELAY_GROUP']>0).sort('DEP_DELAY_GROUP'))

# COMMAND ----------

## Compare how many flights are delayed vs not delayed vs missing information

## In this cell, delays are based on departure time
import matplotlib.ticker as ticker

# Aggregate the flights data
delayed_flights = df.groupby('DEP_DELAY_GROUP').count().filter(df['DEP_DELAY_GROUP']>0).sort('DEP_DELAY_GROUP').toPandas()

# Plot the flight status
fig, ax = plt.subplots(figsize=(16,8))

fig1 = plt.bar(x = delayed_flights["DEP_DELAY_GROUP"], height = delayed_flights["count"])
plt.xlabel("Flight Delay Group", fontsize = 12)
plt.ylabel("Number of Flights", fontsize = 12)
plt.title("Distribution of delayed flights by delay group", fontsize = 14)

ax.bar_label(fig1, 
             labels = [('%.2f' % val) + '%' for val in delayed_flights["count"] / sum(delayed_flights["count"])*100], 
             label_type = 'edge', padding = 2);
# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["count"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Delayed flights by ORIGIN airport

# COMMAND ----------

display(df.filter(df['DEP_DEL15']==1).groupby('ORIGIN').count().sort('count', ascending=False).take(20))

# COMMAND ----------

## Compare how many flights are delayed vs not delayed vs missing information

## In this cell, delays are based on departure time
import matplotlib.ticker as ticker

# Aggregate the flights data
delayed_flights = df.filter(df['DEP_DEL15']==1).groupby('ORIGIN').count().sort('count', ascending=False).toPandas().head(20)

# Plot the flight status
fig, ax = plt.subplots(figsize=(16,8))

fig1 = plt.bar(x = delayed_flights["ORIGIN"], height = delayed_flights["count"])
plt.xlabel("Airport", fontsize = 12)
plt.ylabel("Number of Flights", fontsize = 12)
plt.title("Number of delays by ORIGIN airport", fontsize = 14)

ax.bar_label(fig1, 
             labels = [('%.2f' % val) + '%' for val in delayed_flights["count"] / sum(delayed_flights["count"])*100], 
             label_type = 'edge', padding = 2);
# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["count"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delayed flights by DESTINATION airport

# COMMAND ----------

display(df.filter(df['DEP_DEL15']==1).groupby('DEST').count().sort('count', ascending=False).take(20))

# COMMAND ----------

import matplotlib.ticker as ticker

# Aggregate the flights data
delayed_flights = df.filter(df['DEP_DEL15']==1).groupby('DEST').count().sort('count', ascending=False).toPandas().head(20)

# Plot the flight status
fig, ax = plt.subplots(figsize=(16,8))

fig1 = plt.bar(x = delayed_flights["DEST"], height = delayed_flights["count"])
plt.xlabel("Airport", fontsize = 12)
plt.ylabel("Number of Flights", fontsize = 12)
plt.title("Number of delays by DESTINATION airport", fontsize = 14)

ax.bar_label(fig1, 
             labels = [('%.2f' % val) + '%' for val in delayed_flights["count"] / sum(delayed_flights["count"])*100], 
             label_type = 'edge', padding = 2);
# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["count"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Delayed flights by ORIGIN-DESTINATION pair

# COMMAND ----------

display(df.filter(df['DEP_DEL15']==1).groupby('ORIGIN','DEST').count().sort('count', ascending=False).withColumn('ORIGIN-DESTINATION', concat(col('ORIGIN'), lit("-"), col("DEST"))).take(20))

# COMMAND ----------

import matplotlib.ticker as ticker

# Aggregate the flights data
delayed_flights = df.filter(df['DEP_DEL15']==1).groupby('ORIGIN','DEST').count().sort('count', ascending=False).withColumn('ORIGIN-DESTINATION', concat(col('ORIGIN'), lit("-"), col("DEST"))).toPandas().head(20)
delayed_flights.set_index('ORIGIN','DEST')
# Plot the flight status
fig, ax = plt.subplots(figsize=(20,10))

fig1 = plt.bar(x = delayed_flights['ORIGIN-DESTINATION'], height = delayed_flights["count"])
plt.xlabel("Airport Pair", fontsize = 12)
plt.ylabel("Number of Flights", fontsize = 12)
plt.title("Number of delays by ORIGIN-DESTINATION airport pair", fontsize = 14)
plt.xticks(rotation=45)

ax.bar_label(fig1, 
             labels = [('%.2f' % val) + '%' for val in delayed_flights["count"] / sum(delayed_flights["count"])*100], 
             label_type = 'edge', padding = 2);
# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["count"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delays by Airline

# COMMAND ----------

display(df.filter(df['DEP_DEL15']==1).groupby('Air Carrier Name').count().sort('count', ascending=False))

# COMMAND ----------

import matplotlib.ticker as ticker

# Aggregate the flights data
delayed_flights = df.filter(df['DEP_DEL15']==1).groupby('Air Carrier Name').count().sort('count', ascending=False).toPandas()

# Plot the flight status
fig, ax = plt.subplots(figsize=(20,8))

fig1 = plt.bar(x = delayed_flights["Air Carrier Name"], height = delayed_flights["count"])
plt.xlabel("Airline", fontsize = 12)
plt.ylabel("Number of Flights", fontsize = 12)
plt.title("Number of delayed flights by airline", fontsize = 14)
plt.xticks(rotation=90)

ax.bar_label(fig1, 
             labels = [('%.2f' % val) + '%' for val in delayed_flights["count"] / sum(delayed_flights["count"])*100], 
             label_type = 'edge', padding = 2);
# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["count"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Frequency Analysis

# COMMAND ----------

# Total number of flights by month, by year
df_mon_yr = df.dropna(subset='DEP_DEL15').groupBy('YEAR', 'MONTH').count().sort('YEAR', 'MONTH', ascending=True)


# COMMAND ----------

flights_2015 = df_mon_yr[(df_mon_yr.YEAR == "2015")].toPandas()
flights_2016 = df_mon_yr[(df_mon_yr.YEAR == "2016")].toPandas()
flights_2017 = df_mon_yr[(df_mon_yr.YEAR == "2017")].toPandas()
flights_2018 = df_mon_yr[(df_mon_yr.YEAR == "2018")].toPandas()
flights_2015.plot(kind='line',x='MONTH',y='count',color='red')
flights_2016.plot(kind='line',x='MONTH',y='count',color='blue')
flights_2017.plot(kind='line',x='MONTH',y='count',color='orange')
flights_2018.plot(kind='line',x='MONTH',y='count',color='green')
plt.title('Monthly flights by year')
plt.xlabel('Month')
plt.ylabel('Number of flights')
plt.legend(['year 2015', 'year 2016', 'year 2017', 'year 2018'], loc=4)
plt.show()

# COMMAND ----------

# Total number of flights by airport
delay_summary = df.dropna(subset='DEP_DEL15').groupBy('ORIGIN', 'DEST', 'Air Carrier Name', 'DEP_DEL15').count().sort('count', ascending=False)
delay_summary = delay_summary.alias('df_1').join(delay_summary.alias('df_2'), \
                               (col('df_1.ORIGIN') == col('df_2.ORIGIN')) & (col('df_1.DEST') == col('df_2.DEST')) & (col('df_1.Air Carrier Name') == col('df_2.Air Carrier Name')) & (col('df_1.DEP_DEL15') == 0) & (col('df_2.DEP_DEL15') == 1)) \
        .select(col('df_1.ORIGIN'), col('df_1.DEST'), col('df_1.Air Carrier Name'), \
                col('df_1.count').alias('ontime_count'), col('df_2.count').alias('delay_count'))
delay_summary = delay_summary.withColumn('total', delay_summary.ontime_count + delay_summary.delay_count)
delay_summary = delay_summary = delay_summary.withColumn('percentage_delay', delay_summary.delay_count/delay_summary.total).withColumn('ORIGIN-DESTINATION', concat(col('ORIGIN'), lit("-"), col("DEST")))
display(delay_summary.sort('percentage_delay', ascending=False))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Frequency of delays by ORIGIN airport

# COMMAND ----------

origin_delay = delay_summary.groupBy('ORIGIN').sum('ontime_count', 'delay_count', 'total')
origin_delay = origin_delay.withColumn('percentage_delay', origin_delay['sum(delay_count)']/origin_delay['sum(total)'])
display(origin_delay.where(origin_delay['sum(total)']>10000).sort('percentage_delay', ascending=False))

# COMMAND ----------

## Delays by Origin 

## In this cell, delays are based on departure time
import matplotlib.ticker as ticker

# Aggregate the flights data
# delayed_flights = df.filter(df['DEP_DEL15']==1).groupby('ORIGIN').count().sort('count', ascending=False).toPandas().head(20)
delayed_flights = origin_delay.where(origin_delay['sum(total)']>10000).sort('percentage_delay', ascending=False).toPandas().head(20)

# Plot the flight status
fig, ax = plt.subplots(figsize=(16,8))

fig1 = plt.bar(x = delayed_flights["ORIGIN"], height = delayed_flights["percentage_delay"])
plt.xlabel("Airport", fontsize = 12)
plt.ylabel("Number of Flights", fontsize = 12)
plt.title("Frequency of delays by ORIGIN airport", fontsize = 14)

ax.bar_label(fig1, 
             labels = [('%.2f' % val) + '%' for val in delayed_flights["percentage_delay"]], 
             label_type = 'edge', padding = 2);
# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["percentage_delay"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Frequency of delays by DESTINATION airport

# COMMAND ----------

dest_delay = delay_summary.groupBy('DEST').sum('ontime_count', 'delay_count', 'total')
dest_delay = dest_delay.withColumn('percentage_delay', dest_delay['sum(delay_count)']/dest_delay['sum(total)'])
display(dest_delay.where(dest_delay['sum(total)']>10000).sort('percentage_delay', ascending=False))

# COMMAND ----------

## Delays by Destination

## In this cell, delays are based on departure time
import matplotlib.ticker as ticker

# Aggregate the flights data
# delayed_flights = df.filter(df['DEP_DEL15']==1).groupby('ORIGIN').count().sort('count', ascending=False).toPandas().head(20)
delayed_flights = dest_delay.where(dest_delay['sum(total)']>10000).sort('percentage_delay', ascending=False).toPandas().head(20)

# Plot the flight status
fig, ax = plt.subplots(figsize=(16,8))

fig1 = plt.bar(x = delayed_flights["DEST"], height = delayed_flights["percentage_delay"])
plt.xlabel("Airport", fontsize = 12)
plt.ylabel("Number of Flights", fontsize = 12)
plt.title("Frequency of delays by DESTINATION airport", fontsize = 14)

ax.bar_label(fig1, 
             labels = [('%.2f' % val) + '%' for val in delayed_flights["percentage_delay"]], 
             label_type = 'edge', padding = 2);
# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["percentage_delay"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Frequency of delays by airport pair

# COMMAND ----------

airport_pairs = delay_summary.groupBy('ORIGIN','DEST').sum('ontime_count', 'delay_count', 'total')
airport_pairs = airport_pairs.withColumn('percentage_delay', airport_pairs['sum(delay_count)']/airport_pairs['sum(total)'])
display(airport_pairs.where(airport_pairs['sum(total)'] > 8000).sort('percentage_delay', ascending=False))

# COMMAND ----------

import matplotlib.ticker as ticker

# Aggregate the flights data
# delayed_flights = df.filter(df['DEP_DEL15']==1).groupby('ORIGIN','DEST').count().sort('count', ascending=False).withColumn('ORIGIN-DESTINATION', concat(col('ORIGIN'), lit("-"), col("DEST"))).toPandas().head(20)
delayed_flights = airport_pairs.where(airport_pairs['sum(total)'] > 8000).sort('percentage_delay', ascending=False).withColumn('ORIGIN-DESTINATION', concat(col('ORIGIN'), lit("-"), col("DEST"))).toPandas().head(20)

delayed_flights.set_index('ORIGIN-DESTINATION')
# Plot the flight status
fig, ax = plt.subplots(figsize=(20,10))

fig1 = plt.bar(x = delayed_flights['ORIGIN-DESTINATION'], height = delayed_flights["percentage_delay"])
plt.xlabel("Airport Pair", fontsize = 12)
plt.ylabel("Number of Flights", fontsize = 12)
plt.title("Frequency of delays by ORIGIN-DESTINATION airport pair", fontsize = 14)
plt.xticks(rotation=45)

ax.bar_label(fig1, 
             labels = [('%.2f' % val) + '%' for val in delayed_flights["percentage_delay"]], 
             label_type = 'edge', padding = 2);
# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["percentage_delay"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Frequency of delays by airline

# COMMAND ----------

airlines_delay = delay_summary.groupBy('Air Carrier Name').sum('ontime_count', 'delay_count', 'total')
airlines_delay = airlines_delay.withColumn('percentage_delay', airlines_delay['sum(delay_count)']/airlines_delay['sum(total)'])
display(airlines_delay.sort('percentage_delay', ascending=False))

# COMMAND ----------

import matplotlib.ticker as ticker

# Aggregate the flights data
# delayed_flights = df.filter(df['DEP_DEL15']==1).groupby('Air Carrier Name').count().sort('count', ascending=False).toPandas()
delayed_flights = airlines_delay.sort('percentage_delay', ascending=False).toPandas()

# Plot the flight status
fig, ax = plt.subplots(figsize=(20,8))

fig1 = plt.bar(x = delayed_flights["Air Carrier Name"], height = delayed_flights["percentage_delay"])
plt.xlabel("Airline", fontsize = 12)
plt.ylabel("Number of Flights", fontsize = 12)
plt.title("Frequency of delayed flights by airline", fontsize = 14)
plt.xticks(rotation=90)

ax.bar_label(fig1, 
             labels = [('%.2f' % val) + '%' for val in delayed_flights["percentage_delay"]], 
             label_type = 'edge', padding = 2);
# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["percentage_delay"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # More data analysis

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Impact of delay of incoming aircraft

# COMMAND ----------

delay_df = df.withColumn('PREV_FLIGHT_DELAY', when(df.LATE_AIRCRAFT_DELAY>0, 1).otherwise(0)).where(df['DEP_DEL15'].isNotNull()).groupBy('DEP_DEL15','PREV_FLIGHT_DELAY').count().withColumnRenamed('count', 'total')

# COMMAND ----------

display(delay_df)

# COMMAND ----------

# What is the probability that a flight is delayed if the incoming aircraft is late?

delay_df_1 = delay_df.where(delay_df.PREV_FLIGHT_DELAY==1)
delay_df_1 = delay_df_1.crossJoin(delay_df_1.select(sum('total').alias('sum_total'))).withColumn('percentage', col('total') / col('sum_total'))
display(delay_df_1)

# COMMAND ----------

# What is the probability that a flight is delayed if the incoming aircraft is on time?

delay_df_1 = delay_df.where(delay_df.PREV_FLIGHT_DELAY==0)
delay_df_1 = delay_df_1.crossJoin(delay_df_1.select(sum('total').alias('sum_total'))).withColumn('percentage', col('total') / col('sum_total'))
display(delay_df_1)

# COMMAND ----------

# Given a flight is delayed, what is the probability that the incoming aircraft was late?

delay_df_1 = delay_df.where(delay_df.DEP_DEL15==1)
delay_df_1 = delay_df_1.crossJoin(delay_df_1.select(sum('total').alias('sum_total'))).withColumn('percentage', col('total') / col('sum_total'))
display(delay_df_1)

# COMMAND ----------

# Given a flight is on time, what is the probability that the incoming aircraft was late?

delay_df_1 = delay_df.where(delay_df.DEP_DEL15==0)
delay_df_1 = delay_df_1.crossJoin(delay_df_1.select(sum('total').alias('sum_total'))).withColumn('percentage', col('total') / col('sum_total'))
display(delay_df_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Flight durations

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.select('CRS_ELAPSED_TIME').describe())

# COMMAND ----------

display(df.where(df['CRS_ELAPSED_TIME'].isNull()))

# COMMAND ----------

display(df.where(df['CRS_ELAPSED_TIME']<=0))

# COMMAND ----------

distance_df = df.where((df['CRS_ELAPSED_TIME']>0) & (df['CRS_ELAPSED_TIME'].isNotNull()))

# COMMAND ----------

distance_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Tail Number EDA

# COMMAND ----------

tail_num_filepath = "dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/new_data/aircrafts/mapped_tail_num.csv"

tail_num_data = spark.read.options(inferSchema = True, header = True).csv(tail_num_filepath)
display(tail_num_data)

# COMMAND ----------

df_depart = df.select('DEP_DEL15', 'TAIL_NUM', 'YEAR', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DISTANCE_GROUP')
display(df_depart)

# COMMAND ----------

df_tail_num = df_depart.join(tail_num_data, (df_depart.TAIL_NUM == tail_num_data.tail_num) & 
                             (df_depart.YEAR == tail_num_data.year),
                             "left")
display(df_tail_num)

# COMMAND ----------

# Aggregate the flights data
delayed_flights = df_tail_num.groupby('year_mfr').mean("DEP_DEL15").sort("year_mfr", ascending=True).toPandas()
delayed_flights = delayed_flights.dropna()


# Plot the flight status
fig, ax = plt.subplots(figsize=(20,8))

fig1 = plt.bar(x = delayed_flights["year_mfr"], height = delayed_flights["avg(DEP_DEL15)"])
plt.xlabel("manufacturing year", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.title("Frequency of Delay by manufacturing year", fontsize = 14)
plt.xticks(rotation=90)


# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["avg(DEP_DEL15)"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------


# Aggregate the flights data
delayed_flights = df_tail_num.groupby('mfr').mean("DEP_DEL15").sort("avg(DEP_DEL15)", ascending=False).toPandas().head(20)
delayed_flights = delayed_flights.dropna()


# Plot the flight status
fig, ax = plt.subplots(figsize=(20,8))

fig1 = plt.bar(x = delayed_flights["mfr"], height = delayed_flights["avg(DEP_DEL15)"])
plt.xlabel("manufacturer", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.title("Frequency of Delay by manufacturer", fontsize = 14)
plt.xticks(rotation=90)


# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["avg(DEP_DEL15)"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

delayed_flights = df_tail_num.groupby('model').mean("DEP_DEL15")
display(delayed_flights)

# COMMAND ----------

# Aggregate the flights data
delayed_flights = df_tail_num.groupby('model').mean("DEP_DEL15").sort("avg(DEP_DEL15)", ascending=False).toPandas().head(20)
delayed_flights = delayed_flights.dropna()


# Plot the flight status
fig, ax = plt.subplots(figsize=(20,8))

fig1 = plt.bar(x = delayed_flights["model"], height = delayed_flights["avg(DEP_DEL15)"])
plt.xlabel("Model", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.title("Frequency of Delay by aircraft model", fontsize = 14)
plt.xticks(rotation=90)


# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["avg(DEP_DEL15)"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

# Aggregate the flights data
delayed_flights = df_tail_num.groupby('no-seats').mean("DEP_DEL15").sort("avg(DEP_DEL15)", ascending=False).toPandas().head(20)
delayed_flights = delayed_flights.dropna()


# Plot the flight status
fig, ax = plt.subplots(figsize=(20,8))

fig1 = plt.bar(x = delayed_flights["no-seats"], height = delayed_flights["avg(DEP_DEL15)"])
plt.xlabel("Number of Seats", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.title("Frequency of Delay by number of seats", fontsize = 14)
plt.xticks(rotation=90)


# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["avg(DEP_DEL15)"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

import matplotlib.ticker as ticker
delayed_flights = df.groupby('DISTANCE_GROUP').mean("DEP_DEL15").sort("avg(DEP_DEL15)", ascending=False).toPandas().head(20)
delayed_flights = delayed_flights.dropna()


# Plot the flight status
fig, ax = plt.subplots(figsize=(20,8))

fig1 = plt.bar(x = delayed_flights["DISTANCE_GROUP"], height = delayed_flights["avg(DEP_DEL15)"])
plt.xlabel("Distance Group", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.title("Frequency of Delay by distance group", fontsize = 14)
plt.xticks(rotation=90)


# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["avg(DEP_DEL15)"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

delayed_flights = df_tail_num.groupby('type-acft').mean("DEP_DEL15").sort("avg(DEP_DEL15)", ascending=False).toPandas().head(20)
delayed_flights = delayed_flights.dropna()


# Plot the flight status
fig, ax = plt.subplots(figsize=(20,8))

fig1 = plt.bar(x = delayed_flights["type-acft"], height = delayed_flights["avg(DEP_DEL15)"])
plt.xlabel("Type of Aircraft", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.title("Frequency of Delay by Type of Aircraft", fontsize = 14)
plt.xticks(rotation=90)


# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["avg(DEP_DEL15)"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------



delayed_flights = df_tail_num.groupby('ac-weight').mean("DEP_DEL15").sort("avg(DEP_DEL15)", ascending=False).toPandas().head(20)
delayed_flights = delayed_flights.dropna()


# Plot the flight status
fig, ax = plt.subplots(figsize=(20,8))

fig1 = plt.bar(x = delayed_flights["ac-weight"], height = delayed_flights["avg(DEP_DEL15)"])
plt.xlabel("Aircraft Weight", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.title("Frequency of Delay by Aircraft Weight", fontsize = 14)
plt.xticks(rotation=90)


# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["avg(DEP_DEL15)"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

delayed_flights = df_tail_num.groupby('no-eng').mean("DEP_DEL15").sort("avg(DEP_DEL15)", ascending=False).toPandas().head(20)
delayed_flights = delayed_flights.dropna()


# Plot the flight status
fig, ax = plt.subplots(figsize=(20,8))

fig1 = plt.bar(x = delayed_flights["no-eng"], height = delayed_flights["avg(DEP_DEL15)"])
plt.xlabel("Number of Engines", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.title("Frequency of Delay by Number of Engines", fontsize = 14)
plt.xticks(rotation=90)


# make figure taller to fit percent label
ax.set_ylim(top = max(delayed_flights["avg(DEP_DEL15)"])*1.1);
# fix y-tick values
ax.get_xticks()
# ax.yaxis.set_major_locator(ticker.FixedLocator(delayed_flights["count"]))
# ax.set_yticklabels(['{:.0f}'.format(x) for x in delayed_flights["count"]])
fig.show()

# COMMAND ----------

display(df_tail_num.groupby('no-eng').count())

# COMMAND ----------

display(df_tail_num.groupby('no-seats').count())

# COMMAND ----------

display(df_tail_num.groupby('year_mfr').count())

# COMMAND ----------

display(df_tail_num.groupby('mfr').count())

# COMMAND ----------

display(df_tail_num.groupby('ac-weight').count())

# COMMAND ----------

display(df_tail_num.groupby('type_aircraft').count())

# COMMAND ----------

display(df_tail_num.groupby('type_engine').count())

# COMMAND ----------

df_tail_num = df_tail_num.withColumn('year_mfr', col('year_mfr').cast('double'))
corr_cols = ['DEP_DEL15', 'CRS_ELAPSED_TIME', 'DISTANCE', 'DISTANCE_GROUP', 'year_mfr', 'no-eng', 'no-seats']
sampled_tail_num = df_tail_num.select(*corr_cols).sample(fraction = 0.05).toPandas()
corrs = sampled_tail_num.corr()

# vectorCol = 'corr_features'
# # df_tail_num.select(*cols_to_vector).corr()
# assembler = VectorAssembler().setInputCols(cols_to_vector).setOutputCol(vectorCol)
# df_vector = assembler.transform(df_tail_num).select(vectorCol)
# matrix = Correlation.corr(df_vector, vectorCol)
# cor_np = matrix.collect()[0][matrix.columns[0]].toArray()

# COMMAND ----------

cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corrs, 
            cmap = cmap,
            xticklabels=sampled_tail_num.columns,
            yticklabels=sampled_tail_num.columns)
plt.title('Correlation Matrix with Tail Num Data');

# COMMAND ----------


