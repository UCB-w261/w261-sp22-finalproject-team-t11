# Databricks notebook source
# MAGIC %md
# MAGIC # Flight Delay Predictor
# MAGIC 
# MAGIC ## W261 Final Project
# MAGIC ## Section 2 - Team 11
# MAGIC 
# MAGIC ### Jeffrey Adams, Jenny Conde, Dante Malagrino, Joy Moglia, Ram Senthamarai

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Notebook Setup

# COMMAND ----------

!pip install geopandas

# COMMAND ----------

!pip install holidays

# COMMAND ----------

import os
from pyspark import SparkFiles
from pyspark.sql.functions import col, split, to_utc_timestamp, count
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import geopandas as gpd
from datetime import date, timedelta
import seaborn as sns
import holidays

%matplotlib inline

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
AIRPORTS_MASTER_LOC = f"{blob_url}/staged/airports"
AIRPORTS_WS_LOC = f"{blob_url}/staged/airports_weatherstations"
SHAPES_BASE_FOLDER = "/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/shapes"

# COMMAND ----------

# MAGIC %md
# MAGIC # 1 Phase I Summary
# MAGIC 
# MAGIC ### Past - What we completed in this phase:
# MAGIC 
# MAGIC  - Key questions and goals
# MAGIC    - Binary delay/no-delay classification for departing flights with 2-hour advance notice
# MAGIC    - What are the key factors that cause flight delays?
# MAGIC  - Model performance metrics
# MAGIC    - Chose \\(F_{\beta}\\) with \\(\beta=0.5\\) to prioritize precision vs. recall (i.e. we will build a model that favors false negatives vs. false positives)
# MAGIC  - EDA:
# MAGIC    - Unbalanced dataset (~75% on-time flights)
# MAGIC    - Flights are delayed more in the later hours of the day
# MAGIC  - Potential *new* features to include:
# MAGIC    - Holidays
# MAGIC    - Delay of incoming flight
# MAGIC    - Type of aircraft (requires a new dataset)
# MAGIC   - Joined datasets
# MAGIC   - Defined Cross-Validation strategy (Blocked CV)
# MAGIC 
# MAGIC ### Problems
# MAGIC  - Time conversions, with daylight savings time taken into cosnideration
# MAGIC  - Mapping of weather stations to airport locations
# MAGIC  - Some tail number don't easily match to aircraft models
# MAGIC  - Weather data not available for 2020/2021 in a consistent format (needs further investigation)
# MAGIC  - High volume of data 
# MAGIC  
# MAGIC ### Plan for Next Phase
# MAGIC    - Resolve 2020 and 2021 weather data issue
# MAGIC    - Finish joining and pre-processing data
# MAGIC    - Split train/validation/test following appropriate methodology for time-series data
# MAGIC    - Stretch goal: Run simple model and derive baseline performance

# COMMAND ----------

# MAGIC %md
# MAGIC # 2 Question Formulation
# MAGIC 
# MAGIC A delayed flight has significant negative implications on passengers, airlines and airports, causing financial and emotional damages to many of the stakeholders involved.  According to a 2008 report by the Joint Economic Committee of the US Senate [1], the total cost of domestic air traffic delays to the U.S. economy in 2007 was as much as $41 billion; and the problem is not *only* economical; according to the same source, delayed flights consumed about 740 million additional gallons of jet fuel, which results in increased greenhouse gas emissions and exacerbates climate change.  Two other interesting statistics from the same report, which are worth mentioning to understand the importance of the problem at hand, are the fact that almost 80% of flight delays in 2007 occurred before take-off (i.e. at departure) and a disproportionate majority of delays were associated to flights originating or landing at the nation's largest airports.
# MAGIC 
# MAGIC Unfortunately, completely avoiding delays can be difficult, if not impossible, since delays are caused by several different sources out of the control of anyone scheduling and operating flights.  A better approach would be to predict delays sufficiently in advance and allow airlines, passengers and airport operators to adjust their plans accordingly.
# MAGIC 
# MAGIC In this proposal, we are taking the perspective of a business that sells prediction services to airline companies to help them predict the potential delay of a flight two hours ahead of the scheduled departure time.  The minimum viable product that defines this service will provide an initial binary result of delay/no-delay, indicating the possibility that a flight would experience a longer than 15 minutes delay at departure.  While we are initially focusing only on airlines, we expect the general results of this modeling work to be applicable to passengers and airports, although the extension of the target audience to those group would require further analysis on key metrics and some of the business decisions that we'll use to guide our development.
# MAGIC 
# MAGIC In summary, the fundamental question we intend to answer is whether a departing flight will be delayed or not by at least 15 minutes, given information available to us 2 hours before the scheduled departure time. A secondary goal is to additionally determine the most important factors that influence whether a flight is delayed or not (e.g. weather conditions, time of day, day of week, is it a holiday).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Dataset
# MAGIC 
# MAGIC The initial dataset used for the development of this product leverages flight and weather data from 2015 to 2019.  We are planning to use these data to train and validate the model. We are planning to test the model against travel data from 2020 and 2021, provided we can identify weather data for that period. The impact of the COVID-19 pandemic on air travel in those years will most likely represent a challenge on the predicting ability of our model, which we intend to evaluate and assess accordingly. Whether a flight is on time or not is determined by a multitiude of factors like weather conditions, existing flight delays, operations, scheduling etc. For this research, we are focusing the factors highlighted in the figure below.
# MAGIC 
# MAGIC ![Causes For Delay](files/shared_uploads/ram.senth@berkeley.edu/features.png)
# MAGIC 
# MAGIC Inspired By: [arXiv:1703.06118](https://arxiv.org/abs/1703.06118) [cs.CY]
# MAGIC 
# MAGIC These features are derived from 4 main datasets. We have the high level overview of the datasets here. More details are in the EDA section below.
# MAGIC 
# MAGIC #### 2.1.1 Flights
# MAGIC We are using a subset of the passenger flight's on-time performance data taken from the TranStats data collection available from the U.S. Department of Transportation (DOT) covering about 30 million flights between 372 airports during the period of 2015-2019.
# MAGIC 
# MAGIC #### 2.1.2 Weather
# MAGIC We are using daily weather data collected from various weather stations published by National Oceanic and Atmospheric Administration repository over the period 2015-2019.
# MAGIC 
# MAGIC #### 2.1.3 Weather Stations
# MAGIC In order for us to focus on weather in and around airports, we use the weather station data which identifies each weather station, its location and proximity to other points of interest. This dataset has about 630 million observations from about 12,000 weather stations over the period of 2015-2019.
# MAGIC 
# MAGIC #### 2.1.4 Airports
# MAGIC The airports data has location, timezone and type information for each airport. This dataset has been sourced from multiple sources:
# MAGIC |Data|Description|
# MAGIC |----|------|
# MAGIC |[Timezone](http://www.fresse.org/dateutils/tzmaps.html)|IANA Timezone code for each airport, used for translating time to match weather data for a diven time.|
# MAGIC |[Basic Information](https://datahub.io/core/airport-codes#resource-airport-codes)|Airport codes in different code schemes, location and type of airport. The codes are used for looking up weather around an airport.|
# MAGIC 
# MAGIC ### 2.1.5 Join Strategy
# MAGIC Identifying the weather around a given airport poses a few different challenges. One is the sheer scale of it. Every station reports at least once an hour, mostly twice. For the 5 year period of our study, we have weather from about 12,000 weather stations and about 31 million flights between 372 airports. We have a lot of data to work with. We plan to use data partioning as a strategy for addressing the scale challenge. We will be partitioning both flights and weather data by date and hour of day. We also plan to limit weather data to about 700 weather stations near the 372 airports for which we have flights information. This will reduce the weather data from 630 million observations to about 40 million observations.
# MAGIC 
# MAGIC Our second challenge is one of data standardization. The weather data and flights data come from two different sources and industries. Hence they follow different standards. For example, the weather data uses ICAO code published by the [International Civil Aviation Organization](https://www.icao.int/Pages/default.aspx) while the flights data uses the more common IATA code published by [Internationa Air Transportation Association](https://en.wikipedia.org/wiki/International_Air_Transport_Association). Similarly, weather data is in UTC timezone while flights data is in local timezone. Our strategy is to use commonly available key datasets for mapping and translating data.
# MAGIC 
# MAGIC Our third challenge is one of data quality. During EDA, we noticed that the weather data from a given station can be unreliable. We noticed that the data is more often marked as low quality or has missing fields. In this case, our strategy is to aggregate data from a few weather stations that are closest to the airport, ignoring low quality data. We used [Ball Tree](https://en.wikipedia.org/wiki/Ball_tree) based neighbor search algorithm to identify upto 5 weather stations within a 20 mile radius of each airport.
# MAGIC 
# MAGIC The following diagram shows how all these strategies come together for generating the required features.
# MAGIC 
# MAGIC ![Join Strategy](files/shared_uploads/ram.senth@berkeley.edu/join_strategy-1.png)
# MAGIC 
# MAGIC We will be using the following pipeline architecture for these and other transformations.
# MAGIC 
# MAGIC ![Data Pipeline](files/shared_uploads/ram.senth@berkeley.edu/data_pipeline.png)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 2.2 Metrics
# MAGIC Paraphrasing a quote commonly used in the business world, "*if you can't measure it, it doesn't perform*." In the context of this model, we are going to build a classifier that will determine whether a flight will be delayed or not.  As with all classifiers, we have to spend a little time defining the metrics that will be more interesting for us, especially considering the fact that we are working with a skewed dataset, which over-represents the class of on-time flights.  Let's start by redefining some basic concepts so that we can use a common terminology throughout this analysis.  If we define: \\(TP\\) as the number of \\(True Positives\\), \\(TN\\) as the number of \\(True Negatives\\), \\(FP\\) as the number of \\(False Positives\\) and \\(FN\\) as the number of \\(False Negatives\\), we can look at the key evaluation metrics in the following way:
# MAGIC 
# MAGIC \\[Accuracy =  \dfrac{TP +TN}{TP + FP + TN + FN}\\]
# MAGIC 
# MAGIC \\[Precision =  \dfrac{TP}{TP + FP}\\]
# MAGIC 
# MAGIC \\[Recall =  \dfrac{TP}{TP + FN}\\]
# MAGIC 
# MAGIC Accuracy is the simplest way to evaluate a classification model and does a good job of providing an initial sense of the quality of the model.  Yet, it is a relatively generic metric that does not help tune the model for specific business requirements.
# MAGIC In our case, we are planning to introduce a tool that would help improve the current state of the airline industry by providing customers (airlines or passengers) with a predicted delay for a flight that is not scheduled to depart for another 2 hours.  With this goal, while we believe that a false negative (i.e. we predict no-delay, but the flight gets delayed) is undesirable, we also acknowledge that it would not be worse than the status quo: it would cause disappointment for customers, but no further damage.  On the other hand, a false positive (i.e. we predict a delay and the flight departs on-time) might cause passengers and airlines to unnecessarily replan around the wrongly predicted flight delay; in practical terms, this may cause the airline to scramble to find an alternative aircraft to complete the flight or passengers to choose alternative ways of transportation that were less convenient to begin with.
# MAGIC 
# MAGIC Given the above considerations, but still wanting to achieve a balance between \\(Precision\\) and \\(Recall\\), we have chosen to use the \\(F_{\beta}\\) metric to measure the performance of our model:
# MAGIC 
# MAGIC \\[F_\beta=   \dfrac{(1 + \beta^2) \cdot (Precision \cdot Recall) }{ (\beta^2 \cdot Precision + Recall)}\\]  
# MAGIC 
# MAGIC Consistently with the business implications discussed above, we have chosen a \\(\beta = 0.5\\), which balances the \\(Precision\\) and \\(Recall\\) factors, while giving twice as much weight to \\(Precision\\) than \\(Recall\\). Our choice of metric was also influenced by the inherent imbalance in the datase - only about 20% of flights are delayed.

# COMMAND ----------

# MAGIC %md
# MAGIC # 3 Exploratory Data Analysis and Discussion of Challenges
# MAGIC 
# MAGIC We conducted an exploratory data analysis (EDA) on each of the datasets described above and have selected a subset of our EDA to present in this analysis. In our EDA process, we began with subsets of the airlines dataset and weather dataset. Each subset contained data points from the first quarter of 2015. We recognize that this subset of data is not appropriate for building a model to predict all flights, since there would be high bias. However, the subsets helped us get an initial understanding of the data, and we have included some findings about each dataset below.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 3.1 Airlines Dataset
# MAGIC 
# MAGIC Our "Airlines Dataset" contains information about each flight recorded in the U.S. Department of Transportation's database [[2]](https://www.transtats.bts.gov/homepage.asp). While the full dataset contains information from 2015 to 2019, we conducted our initial EDA on the first quarter of 2015 and further limited the data to only two airports: Chicago O'Hare (ORD) and Atlanta (ATL). This subset of data contains about 160,000 flights, while the entire dataset contains over 31 million flights. The data contains 109 features, which can be broadly grouped into eleven categories:
# MAGIC 1. Time Period
# MAGIC 2. Airline
# MAGIC 3. Origin
# MAGIC 4. Destination
# MAGIC 5. Departure Performance
# MAGIC 6. Arrival Performance
# MAGIC 7. Cancellations & Diversions
# MAGIC 8. Flight Summaries
# MAGIC 9. Cause of Delay
# MAGIC 10. Gate Return Information at Origin Airport
# MAGIC 11. Diverted Airport Information
# MAGIC 
# MAGIC From the set of features, we will be using `DEP_DEL15` as our outcome variable. This feature is an indicator that equals 1 when a flight has a delay greater than 15 minutes.
# MAGIC 
# MAGIC We were able to determine the meaning of each feature and decide each feature's relevance for our analysis and for developing a predictive model. For instance,  features in the "Cause of Delay" category should not be included in our model since these causes are determined after a delay is already known. However, we can analyze the frequency of delay causes to help us determine which other features should be prioritized for our model. 
# MAGIC 
# MAGIC Another interesting finding from our EDA is that each flight is duplicated in our dataset. Because of this, we will need to remove duplicates from our final, clean dataset. The original size of the entire dataset (not limited to ORD/ATL in Q1 2015) was 62,513,788, and once the duplicates are removed, this matches the number of flights we would expect.

# COMMAND ----------

# DBTITLE 1,Flights Table / Airlines Dataset (1-Quarter)
# Load 2015 Q1 for Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data_3m/")
display(df_airlines)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The bar chart below displays the frequency of our outcome variable. About 74% of flights are not delayed by an amount greater than 15 minutes, about 23% of flights are delayed, and 3% of the data is missing. This information is helpful since it shows our target classes are unbalanced, which will play a key role in how we develop our model and interpret the results. For our goal of performing a multinomial classification, we also analyzed the degree to which flights are delayed. There is a large skew in the data--most departing flights are either not delayed or leave early, and there is a long right tail as delay time increases.

# COMMAND ----------

## Compare how many flights are delayed vs not delayed vs missing information
## In this cell, delays are based on departure time

# Aggregate the flights data
delayed_flights = df_airlines.groupBy("DEP_DEL15").count()

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

# MAGIC %md We believe that if an airport has historically been susceptible to flight delays, it is likely to be an important feature in our model. We can create a variable that represents proportion of delayed flights by origin airport and use this as a feature.

# COMMAND ----------

# create new column that shows average delay for the origin airport

avedelay_by_origin = df_airlines.groupBy('ORIGIN').mean("DEP_DEL15").sort("avg(DEP_DEL15)", ascending=False).toPandas()
avedelay_by_origin.set_index('ORIGIN', inplace=True)

def _helper(x):
  y = avedelay_by_origin.loc[x]["avg(DEP_DEL15)"]
  return float(y)

helperudf = udf(lambda x: _helper(x))

df_airlines = df_airlines.withColumn('ORIGIN_AVG_DELAY', helperudf(col('ORIGIN')))
display(df_airlines)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We also suspect that flights during the holidays might suffer from delays more frequently than non-holidays due to more people traveling. We added 2 new columns called 'Holiday' and 'Holiday_5Day' that indicate if the departure date falls on a holiday or within 5 days of a holiday. The code is below.

# COMMAND ----------

# create list of all US holidays
us = holidays.US()

# create a new column called 'Holiday': (1=yes, 0=no)
def _helper(x):
  year, month, day = x.split('-')
  _date = date(year=int(year), month=int(month), day=int(day))
  # returns 1 if x is a US holiday, 0 if not
  return (_date in us)*1

_helperudf = udf(lambda x: _helper(x))

df_airlines = df_airlines.withColumn('Holiday', _helperudf(col('FL_DATE')))


# create a new column 'Holiday_window' if departure date is within 5 days of a holiday: (1=yes, 0=no)
def _helper(x):
  year, month, day = x.split('-')
  _date = date(year=int(year), month=int(month), day=int(day))
  for delta in range(-5,6):
    if _date + timedelta(days=delta) in us:
      return 1
  return 0

_helperudf = udf(lambda x: _helper(x))

df_airlines = df_airlines.withColumn('Holiday_5Day', _helperudf(col('FL_DATE')))
display(df_airlines)

# COMMAND ----------

## Compare the extent to which flights are delayed for both departures and arrivals
## **we don't need this column because we have DepDel15 which already tells us if a flight is delayed or not**

## -2: delay < -15 min
## -1: Delay between -15 and -1 minutes
## 0: Delay between 0 and 14 minutes
## 1: Delayed btween 15 and 29 minutes   **flights 1 and greater is delayed, which is reflected in DepDel15

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
# MAGIC We also suspected that earlier flights might be less delayed than later flights. The data suggests this is true.  

# COMMAND ----------

# look at average # flights that are delayed by hour of day
# flights later in the day tend to be more delayed than in the morning

def bucket(x):
  #returns the hour of departure
  x=int(x)
  x=int(x/100)
  return x

bucketudf = udf(lambda x: bucket(x))

df_delaybyhour = df_airlines.withColumn('HOUR', bucketudf(col('CRS_DEP_TIME'))).groupby('HOUR').mean("DEP_DEL15").sort("HOUR", ascending=False).toPandas()

# change hour to float
df_delaybyhour['HOUR'] = df_delaybyhour['HOUR'].astype(float)

# Plotting average # flights delayed by hour in day
df_delaybyhour.plot.scatter(x='HOUR', y='avg(DEP_DEL15)');
plt.ylabel('Proportion of Delayed Flights');
plt.title('Hour of day is positively correlated with flight delay frequency');

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC These graphs above are a small subset of the EDA conducted on this dataset. We additionally analyzed the relationship between flight delays and day of week, month of year, quarter (which can be interpreted as a proxy for season), and airline. All of these qualitative variables show variation in their correlation to flight delay and will be worthwhile including in our initial prediction model.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 3.2 Weather Stations

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The second dataset in our analysis, "Weather Stations," includes information about weather stations in the U.S. This dataset is much smaller than the flights dataset (and the weather dataset, which will be discussed next). The primary purpose of this dataset is to help us connect the departing airport for each flight to the correct weather station. This dataset typically associates weather stations with their closest airport and subsequently identifies airports using their ICAO codes. The flights dataset uses the IATA codes standard, so we used an intermediate dataset to convert from ICAO codes to IATA codes. Once we had the IATA codes for the airports in the weather stations dataset, we found that there were eight airports missing from the weather stations dataset: PSE, PPG, OGS, SPN, SJU, TKI, GUM, and XWA. Of these, SJU and GUM are large airports. Because these airports are missing from the dataset, we will have to associate weather for flights from these airports using other closeby weather stations. More details on this are included later in the "Data Preparation" section.

# COMMAND ----------

# DBTITLE 1,Weather Stations
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
display(df_stations)

# COMMAND ----------

# MAGIC 
# MAGIC %md
# MAGIC 
# MAGIC ## 3.3 Weather

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The next dataset in our analysis is the "Weather" dataset, which contains global weather information from 2015 to 2019. The full dataset contains over 630 million rows and 177 features. We were able to narrow down this dataset to the correct locations by matching the station IDs from the weather stations dataset with the station IDs in the weather dataset. We additionally limited the features to those that seemed most relevant for this analysis, including wind (speed, direction), temperature, dew point, weather ceiling, horizontal visibility, sea-level pressure, precipitation, and miscellaneous daily weather conditions (e.g. fog, thunder, tornado, and snow). With a limited dataset (which was still quite large), we were able to sample the data and produce visualizations to better understand the distributions of these features.

# COMMAND ----------

# DBTITLE 1,Weather Table (1-Quarter)
# Import the weather data, only showing the stations that we care about
df_weather = spark.read.load("dbfs:/FileStore/shared_uploads/jconde@berkeley.edu/weather_filtered.csv", header=True, format='csv')

# Load the 2015 Q1 for Weather
#df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*").filter(col('DATE') < "2015-04-01T00:00:00.000")
display(df_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC One example of the EDA we conducted on our weather features includes Wind Speed. Wind Speed includes both the speed itself and the quality of the speed. We found that about 97% pass all quality control checks, just under 3% of data points are missing, and the remaining values are likely suspect or erroneous. When we remove the missing data, we can visualize the distribution of wind speed. From the graphs below, we can see that wind speed follows a bimodal distribution and has a long right tail. Wind speed also tends to increase during later hours of the night. We should consider both of these factors when using wind speed as a predictor in our model.

# COMMAND ----------

# Get random sample of weather data to visualize
sampled_weather = df_weather.sample(fraction = 0.05)
sampled_weather_pd = sampled_weather.toPandas()
sampled_weather_pd['WND_Speed'] = sampled_weather_pd['WND_Speed'].astype(float)
sampled_weather_pd['date'] = pd.to_datetime(sampled_weather_pd['date'])
# sampled_weather_pd

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
# MAGIC ## 3.4 Data Preparation

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4.1 Data Wrangling

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.4.1.1 Weather
# MAGIC 
# MAGIC We had to conduct significant data cleaning on the weather dataset in order to get it in a usable form. As mentioned earlier, we first reduced the size of the dataset by filtering only to the weather stations either at airports or closeby to airports for the airports without a weather station on site (see more information about this below and in the Join Strategy section). Then, we identified and selected relevant variables beyond the station identifiers, which included wind, ceiling height, visibility, temperature, sea-level pressure, dew point, precipitation, and daily weather conditions. Each of these features contained subfeatures, which needed to be split up and converted into the correct format, such as floating point numbers and applying scaling factors as appropriate. Information about precipitation and daily weather conditions were aggregated across several columns, and we further converted precipitation into an hourly rate (mm/hr) and converted daily weather conditions into 12 one-hot encoded columns. These columns include fog, thunder, ice, dust, tornado, and show.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.4.1.2 Airports and Weather Station
# MAGIC There are eight airports that do not have any weather stations associated directly. Further, the weather data has noise. So, instead of relying on weather data from just one weather station for an airport, we decided to find up to five closest weather stations for each airport and aggregate the data from them. Below chart shows the closest weather stations identified for Chicago O'Haire and the Atlanta airports. The technical details behind how we found the nearest stations are covered in the section on Join Strategy above.

# COMMAND ----------

def plot_many(data):
    # Setup temp view for the airports master data.
    df_airports = spark.read.parquet(AIRPORTS_MASTER_LOC)
    df_airports.createOrReplaceTempView("vw_airports_master")

    num_plots = len(data)
    
    fig, axis = plt.subplots(1, num_plots, figsize=(12, 12), tight_layout=True)

    for i, (state_name, iatas, shapefile, crs) in enumerate(data):
        ax = axis[i]
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{state_name}')
        plot(ax, i, iatas, shapefile, crs)
        
    fig.suptitle('Closest Weather Stations Within 20 Miles', fontsize=16)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.tight_layout()
    plt.show()

        
def plot(ax, index, iatas, shapefile, crs):
    # Load the shape file for the state.
    state = gpd.read_file(f'{SHAPES_BASE_FOLDER}/{shapefile}')
    
    # Set scaling if needed.
    if crs is None:
        state.plot(ax=ax)
    else:
        state.to_crs(epsg=crs).plot(ax=ax)
        
    iatas_str = "'" + "','".join(iatas) + "'"
    
    # Load the airports data and plot it.
    df_airport_locs = spark.sql(f"""
        SELECT iata, name, airport_lat, airport_lon
        FROM vw_airports_master WHERE iata in ({iatas_str})"""
    ).toPandas()
    gdf_airports = gpd.GeoDataFrame(df_airport_locs, geometry=gpd.points_from_xy(df_airport_locs.airport_lon, df_airport_locs.airport_lat))
    gdf_airports.plot(ax=ax, color='red', label=df_airport_locs['iata'])
    for ind in gdf_airports.index:
        ax.text(df_airport_locs['airport_lon'][ind], df_airport_locs['airport_lat'][ind], s=df_airport_locs['name'][ind], horizontalalignment='left', verticalalignment='top', bbox={'facecolor': 'white', 'alpha':0.2, 'pad': 3, 'edgecolor':'none'})
    
    # Load and plot weather stations data
    df_stations_spark = spark.read.parquet(AIRPORTS_WS_LOC)
    df_stations = df_stations_spark.filter(f"iata in ({iatas_str})").toPandas()
    gdf_ws = gpd.GeoDataFrame(df_stations, geometry=gpd.points_from_xy(df_stations.ws_lon, df_stations.ws_lat))
    gdf_ws.plot(ax=ax, color='lightgreen', alpha=0.5, legend=True)
    

plot_many([('Illinois', ['ORD'], 'illinois_counties/IL_BNDY_County_Py.shp', None), 
           ('Georgia', ['ATL'], 'georgia_counties/GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp', None)])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4.2 Data Split and Cross Validation
# MAGIC 
# MAGIC In this project, we are dealing with time-series data, which require extra care in the definition of the train and test splits.  More specifically, we are considering using Cross Validation as part of the tuning process of the model hyperparameters.  The main challenge with time-series data is to avoid data leakage, which can be caused any time data from the future is used to make a prediction about the presence.  When it comes to Cross Validation, this means that we can't randomly split the the data across multiple sets, but we have to do it in such a way that preserves the sequential order of entries.  More specifically, this means that we have to always test our models with validation and test sets that contain later date flights than those used in the training phase.
# MAGIC 
# MAGIC There are two primary approaches to split time-series data:
# MAGIC  - Time Series Split
# MAGIC  - Blocked Time Series Split
# MAGIC 
# MAGIC The Time Series Split Cross-Validation approach starts by taking a portion of the data and dividing it into two splits (based on a time threshold). The *older* data is used as training set, while the more recent one is used as validation.  Each iteration expands the training set (for example including the portion that was used as validation in the previous iteration) and uses another chunk of data from the portion of the dataset that has not been used, yet, as the new validation set.  This can be repeated until the entire dataset gets used.
# MAGIC 
# MAGIC ![Time Series Split](files/shared_uploads/ram.senth@berkeley.edu/TimeSeries_CV_Split.png)
# MAGIC 
# MAGIC One problem with the Time Series Split is that it can still present some data leakeage as we use the same data for both training and validation at different points in time.  It also has the problem that the datasets being used also has varying sizes at the different stages of the cross validation, so the results of the models should be weighted.
# MAGIC 
# MAGIC The Blocked Cross Validation approach is based on the idea of dividing the entire dataset into multiple chunks, each further divided between a training and validation portion.  By doing this, we avoid memorizing information from an iteration to the next.  The Blocked Cross Validation can be done in two slightly different ways.  The first one is to avoid any overlap between the chunks, while another way to do it would be to have a rolling window over the data and use chunks that could be partially overlapping with each other.  The trade-offs between these two approaches are mostly in terms of complexity of the model being evaluated and computational time required to perform the entire cross validation.
# MAGIC 
# MAGIC ![Blocked Split](files/shared_uploads/ram.senth@berkeley.edu/Blocked_CV_Split-1.png)
# MAGIC 
# MAGIC We will implement a Blocked Time Series Split, but we will evaluate later whether to use completely separate chunks in favor of faster computations or overlapping chunks, which might give us better results.

# COMMAND ----------

# MAGIC %md
# MAGIC # 9. Bibliography and References
# MAGIC 
# MAGIC [1] A Report by the Joint Economic Committee, US Senate, “Your Flight has Been Delayed Again: Flight Delays Cost Passengers, Airlines, and the US Economy Billions”, 2008
# MAGIC https://www.jec.senate.gov/public/index.cfm/democrats/2008/5/your-flight-has-been-delayed-again_1539

# COMMAND ----------


