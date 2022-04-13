# Databricks notebook source
# MAGIC %md
# MAGIC # Flight-XI: A New Flight Delay Predictor
# MAGIC 
# MAGIC #### W261 Final Project (Section 2 - Team 11)
# MAGIC 
# MAGIC * Jeffrey Adams
# MAGIC * Jenny Conde
# MAGIC * Dante Malagrino
# MAGIC * Joy Moglia
# MAGIC * Ram Senthamarai

# COMMAND ----------

import os
from pyspark import SparkFiles
# from pyspark.sql.functions import col, to_utc_timestamp, count, when, year
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configuration_v01 import Configuration
from training_summarizer_v01 import TrainingSummarizer
from down_sampler_v02 import DownSampler
from custom_transformer_v05 import CustomTransformer
import geopandas as gpd

configuration = Configuration()
WEATHER_LOC = f"{configuration.blob_url}/staged/weather"
AIRPORTS_MASTER_LOC = f"{configuration.blob_url}/staged/airports"
SHAPES_BASE_FOLDER = "/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/shapes"
AIRPORTS_WS_LOC = f"{configuration.blob_url}/staged/airports_weatherstations"

%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Introduction
# MAGIC 
# MAGIC You've probably heard of SpaceX, but what about FlightXI? We are a new startup aiming to sell prediction services to airline companies, passengers and airports to help them predict the potential delay of a flight two hours ahead of the scheduled departure time. Our product will provide an initial binary prediction of delay/no-delay, indicating the possibility that a flight would experience a longer than 15 minutes delay at departure.  We are still in our development stages as a stealth startup, but in subsequent parts of the product development, we envision the possibility of implementing a more sophisticated model that would predict the actual extent of the delay within certain buckets (e.g. less than 1 hour, between 1 and 2 hours, longer than 2 hours, etc.).
# MAGIC 
# MAGIC Why focus on flight delays? A delayed flight has significant negative implications on passengers, airlines and airports, causing financial and emotional damages to many of the stakeholders involved.  According to a 2008 report by the Joint Economic Committee of the US Senate [1], the total cost of domestic air traffic delays to the U.S. economy in 2007 was as much as $41 billion. And the problem is not *only* economical; according to the same source, delayed flights consumed about 740 million additional gallons of jet fuel, which results in increased greenhouse gas emissions and exacerbates climate change.  Two other interesting statistics from the same report, which are worth mentioning to understand the importance of the problem at hand, are the fact that almost 80% of flight delays in 2007 occurred before take-off (i.e. at departure) and a disproportionate majority of delays were associated to flights originating or landing at the nation's largest airports.
# MAGIC 
# MAGIC Unfortunately, completely avoiding delays can be difficult, if not impossible, since delays are caused by several different sources out of the control of anyone scheduling and operating flights.  A better approach would be to predict delays sufficiently in advance and allow airlines, passengers and airport operators to adjust their plans accordingly. This is exactly the goal of our startup, and this notebook will walk you, our potential investor, through our model development stages and final results.
# MAGIC 
# MAGIC > Please note that before running code in this notebook, the cells in the <a href="$./init_cluster">init_cluster</a> notebook needs to be run. This is a one time step needed after every cluster restart. The init_cluster script configures Spark context and registers all required custom libraries. Please see **`Code Organization`** section near the end of this notebook for more details.

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Phases Summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Phase 1
# MAGIC ### What We Completed
# MAGIC 
# MAGIC  - Key questions and goals
# MAGIC    - Binary delay/no-delay classification for departing flights with 2-hour advance notice
# MAGIC    - What are the key factors that cause flight delays?
# MAGIC  - Model performance metrics
# MAGIC    - Chose \\(F_{\beta}\\) with \\(\beta=0.5\\) to prioritize precision vs. recall (i.e. we will build a model that favors false negatives vs. false positives)
# MAGIC  - EDA:
# MAGIC    - Flights are delayed more in the later hours of the day
# MAGIC  - Potential *new* features to include:
# MAGIC    - Holiday
# MAGIC    - Delay of incoming flight
# MAGIC    - Type of aircraft (requires a new dataset)
# MAGIC 
# MAGIC ### Problems
# MAGIC  - Time conversions, with daylight savings time taken into cosnideration
# MAGIC  - Mapping of weather stations to airport locations
# MAGIC  - Some tail number don't easily match to aircraft models
# MAGIC  - Weather data not available for 2020/2021 in a consistent format (needs further investigation)
# MAGIC  - High volume of data 
# MAGIC  
# MAGIC  
# MAGIC ### Plan for Next Phase
# MAGIC    - Find 2020 and 2021 weather data
# MAGIC    - Finish joining and pre-processing data
# MAGIC    - Split train/validation/test following appropriate methodology for time-series data
# MAGIC    - Stretch goal: Run simple model and derive baseline performance

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Phase 2
# MAGIC ### What We Completed
# MAGIC 
# MAGIC  - Downloaded and processed the 2020 and 2021 flights data
# MAGIC  - Downloaded and processed 2015-2021 weather data from Local Climatalogical Data
# MAGIC  - Joined all datasets to one master table
# MAGIC  - EDA on final dataset
# MAGIC  - Baseline model
# MAGIC 
# MAGIC ### Challenges And Learnings
# MAGIC 
# MAGIC  - Retrieving and aggregating the data, especially from a new data source
# MAGIC    - Importing about 450 GB of weather data for 7 years and manipulating it was very costly and time consuming, compounded by the inadvertent rookie mistakes. In the process, we learnt that reducing the data work much earlier in the process helps. By identifying the 710 weather stations near the airports of interest and ignoring the remaining 12,000 weather stations even before the ingestion process helped reduce the ingestion time from 5 hours to under 25 minutes.
# MAGIC    - By default Spark was creating multiple 1 mb parquet files that lead to large read times. Repartitioning with year as partition key helped.
# MAGIC    - Spark's datatype inference logic gave us a lot of grief. We ended up ingesting all fields as strings and applied our own conversion for the fields we need. 
# MAGIC  - Optimizing for joining the big datasets
# MAGIC    - Grouping flights and weather into 24 windows per day (1 for the hour of the day), in addition to the reduced weather stations to look at, helped us achieve join time of about 25 minutes. The logic also includes a simple aggregation of weather data from potentially multiple readings within the hour.
# MAGIC  - Identifying potential data leakage issues with current hourly weather aggregation method
# MAGIC  
# MAGIC  
# MAGIC ### Plan for Next Phase
# MAGIC  - Implement train/test split in data pipeline
# MAGIC  - Conduct feature engineering and implement dimensionality reduction
# MAGIC  - Test different models

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 Phase 3
# MAGIC ### What We Completed 
# MAGIC  - We have a fully consolidated dataset that includes flight and weather data from 2015 to 2021
# MAGIC  - We have completed a baseline model with initial results
# MAGIC  - We have a pipeline in place that we can use to experiment and tune models and algorithm
# MAGIC    - This includes implementing cross-validation in ways that take into account the temporal nature of data
# MAGIC    - We are including support for weights and oversampling of the minority class to take into account the imbalanced nature of the dataset
# MAGIC  - We have performed more data analysis and processing to facilitate the feature selection/engineering work that we plan to complete next week
# MAGIC  - We implemented a F_beta metric as part of our CustomCrossValidator
# MAGIC  - We tested methods for addressing imbalance in the data (weighting, oversampling, undersampling)
# MAGIC  
# MAGIC ### Problems
# MAGIC  - Implementing F_beta score with cross validator took more time than expected
# MAGIC  - We obtained aircraft data for tailnumbers but found there's a lot of missing information, and the dataset is unreliable.
# MAGIC  
# MAGIC ### Gap Analysis
# MAGIC 
# MAGIC When comparing our project with others on the leaderboard, we observe several key next steps that we can pursue for our project. Most notably, Team Super Mario’s model has an F_beta score of 0.86, which outperforms our current Logistic Regression baseline model's F_beta score of 0.74 (train on full 2015-2018 data, test on 2019). Team Super Mario uses a Random Forest model to achieve this score, and we plan to implement this model next. Other teams implement additional strategies that we will pursue in the coming week, including hyperparameter tuning, cross validation with multiple folds, and additional feature engineering (ex. PageRank, filling in null values).
# MAGIC 
# MAGIC ### Plan for Next Phase
# MAGIC  - Explore additional models
# MAGIC  - Feature engineering and selection
# MAGIC  - Hyperparameter tuning
# MAGIC  - Test various options for Cross Validation (e.g. ignore the temporal nature of the data an simply run a CrossValidator that treats each line independently of the others)
# MAGIC  - Make final determination on using aircraft data for tailnumbers
# MAGIC  - Determine a preferred model for the final solution and setup for final testing in preparation of the final wrap up

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4 Phase 4
# MAGIC ### What We Completed 
# MAGIC 
# MAGIC * Implemented logistic regression from scratch using gradient descent, optimizing for accuracy but evaluating on \\(F_\beta\\)
# MAGIC * Tried new types of models (Random Forest Classifiers and XGBoost Decision Tree Classifiers).
# MAGIC * Decided on hyper-parameters to optimize as well as cross validation strategy.
# MAGIC * Expanded our feature set and synthesized 3 features (prior flight delay up to 2:15 hours  before  departure, holiday season indicator and delays by airlines) to improve model performance.
# MAGIC * Conducted an analysis on all of our model results, focusing on false positives.
# MAGIC  
# MAGIC ### Problems
# MAGIC 
# MAGIC  - We could not use an \\(F_\beta\\) score in our loss function for gradient descent from scratch since \\(F_\beta\\) is not diferentiable
# MAGIC    - We experimented using Dice loss functions or a smoothed \\(F_\beta\\) score but thought it was best to stick with an accuracy metric as our loss function
# MAGIC  - XGBoost models ran into memory errors
# MAGIC  
# MAGIC ### Gap Analysis
# MAGIC 
# MAGIC There are several techniques that other groups tried to achieve higher scores on different metrics, including \\(F_{1}\\). Team "Friends Across Ocean" had several F scores above 0.82 on both the training and test sets. XGBoost decision trees were used to acheive these F-scores. This is one gap between Friends Across Ocean and us, since we encountered errors when creating XGBoost models. Team Super Mario also achieved a comparable \\(F_{1}\\) score above 0.7 using different hyperparemters than us and also using dimensionality reduction. We could further tune our hyperparameters and explore methods of dimensionality reduction like Principal Component Analysis.
# MAGIC 
# MAGIC ### Plan for Next Phase
# MAGIC  - Present in Live Session

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 2. Question Formulation
# MAGIC 
# MAGIC To translate the overall business problem into a useful predictive model that delivers the outcome just described, we begin by explicitly formulating the question that we intend to answer in the following way:
# MAGIC 
# MAGIC > Will a given flight experience a delay of at least 15 minutes, given information available to us 2 hours before the scheduled departure time?
# MAGIC 
# MAGIC Given available information-- such as weather conditions, time of day, day of weak, holiday season, or other factors-- we will seek to identify the variables most important in influencing flight delay.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Dataset
# MAGIC 
# MAGIC The primary datasets used to develop our product and model are flights and weather data from 2015 to 2021.  The last two years in the data set are most likely anomalies due to the COVID-19 Pandemic and its impact on travel and flights.  Thus, we used 2015-2018 to train and validate our models, while we kept 2019, 2020 and 2021 as test datasets.
# MAGIC 
# MAGIC Whether a flight is on time or not is determined by a multitiude of factors like weather conditions, existing flight delays, operations, and scheduling. For this research, we are focusing on the factors highlighted in the figure below.
# MAGIC 
# MAGIC ![Causes For Delay](files/shared_uploads/ram.senth@berkeley.edu/features.png)
# MAGIC 
# MAGIC Inspired By: [arXiv:1703.06118](https://arxiv.org/abs/1703.06118) [cs.CY]
# MAGIC 
# MAGIC These features are derived from 4 main datasets. We have the high level overview of the datasets here. More details are in the EDA section below.
# MAGIC 
# MAGIC #### 2.1.1 Flights
# MAGIC We are using a subset of the passenger flight's on-time performance data taken from the TranStats data collection available from the U.S. Department of Transportation (DOT) covering about 42.2 million flights between over 370 airports during the period of 2015 to 2021 [[2]](https://www.transtats.bts.gov/homepage.asp).
# MAGIC 
# MAGIC The data contains 109 features, which can be broadly grouped into eleven categories:
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
# MAGIC From the set of features, we will be using `DEP_DEL15` as our outcome variable. This feature is an indicator that equals 1 when a flight has a departure delay greater than 15 minutes and 0 for no delay.
# MAGIC 
# MAGIC We were able to determine the meaning of each feature and decide each feature's relevance for our analysis and for developing a predictive model. For instance,  features in the "Cause of Delay" category should not be included in our model since these causes are determined after a delay is already known. However, we can analyze the frequency of delay causes to help us determine which other features should be prioritized for our model. 
# MAGIC 
# MAGIC #### 2.1.2 Weather
# MAGIC We are using the Local Climatological Data (LCD) dataset published by the National Oceanic and Atmospheric Administration, which contains weather data collected from various weather stations over the period 2015-2021 [[3]](https://www.ncei.noaa.gov/products/land-based-station/local-climatological-data). In order to include the 2020 and 2021 data, we had to switch data sources from our original source, and downloading and ingesting the presented unique challenges in terms of optimizing for scale. **Collecting the 2020 and 2021 data and ultimately testing our model using this data was the primary novelty of our project.** 
# MAGIC 
# MAGIC #### 2.1.3 Weather Stations
# MAGIC In order for us to focus on weather in and around airports, we use the weather station data which identifies each weather station, its location and proximity to other points of interest.
# MAGIC 
# MAGIC #### 2.1.4 Airports
# MAGIC The airports data has location, timezone and type information for each airport. This dataset has been sourced from multiple sources:
# MAGIC |Data|Description|
# MAGIC |----|------|
# MAGIC |[Timezone](http://www.fresse.org/dateutils/tzmaps.html)|IANA Timezone code for each airport, used for translating time to match weather data for a diven time.|
# MAGIC |[Basic Information](https://datahub.io/core/airport-codes#resource-airport-codes)|Airport codes in different code schemes, location and type of airport. The codes are used for looking up weather around an airport.|

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Join Strategy
# MAGIC 
# MAGIC Identifying the weather around a given airport poses a few different challenges. First is the sheer scale of the problem. Each weather station records data at least once per hour, usually twice. For the 7 year period of our study, we have weather from about 12,000 weather stations and about 42 million flights between over 370 airports. To handle this large amount of data, we partitioned both flights and weather data by date and hour of day. We also limited weather data to about 700 weather stations near the 372 airports for which we have flights information. This reduced the weather data from 630 million observations to about 40 million observations. We used [Ball Tree](https://en.wikipedia.org/wiki/Ball_tree) based neighbor search algorithm to identify up to 5 weather stations within a 20-mile radius of each airport to shortlist the 700 weather stations of interest.
# MAGIC 
# MAGIC Our second challenge was standardizing the data. The weather data and flights data come from two different sources and industries. Hence they follow different standards. For example, the weather data uses ICAO code published by the [International Civil Aviation Organization](https://www.icao.int/Pages/default.aspx) while the flights data uses the more common IATA code published by [International Air Transportation Association](https://en.wikipedia.org/wiki/International_Air_Transport_Association). Similarly, weather data is in UTC timezone while flights data is in local timezone. Our strategy is to use commonly available key datasets for mapping and translating data.
# MAGIC 
# MAGIC Our third challenge was data quality. During EDA, we noticed that the weather data from a given station can be unreliable. We noticed that the data is more often marked as low quality or has missing fields. We ignore low quality data. We were planning on aggregating data from a few weather stations that are closest to the airport, ignoring low quality data. However, due to limited time, we couldn't get to it.
# MAGIC 
# MAGIC The following diagram shows how these strategies came together for generating the required features.
# MAGIC 
# MAGIC ![Join Strategy](files/shared_uploads/ram.senth@berkeley.edu/join_strategy-1.png)
# MAGIC 
# MAGIC We used the following pipeline architecture for these and other transformations.
# MAGIC 
# MAGIC ![Data Pipeline](files/shared_uploads/ram.senth@berkeley.edu/data_pipeline.png)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 2.3 Metrics
# MAGIC For this problem, we built a classifier that determines whether a flight will be delayed or not. We intentionally defined our evaluation metric, which took into account our skewed dataset (there are far more on-time flights than delayed flights) as well as severity of incorrect predictions. 
# MAGIC 
# MAGIC We start by redefining some basic concepts to ensure a common terminology throughout this analysis.  If we define: \\(TP\\) as the number of \\(True Positives\\), \\(TN\\) as the number of \\(True Negatives\\), \\(FP\\) as the number of \\(False Positives\\) and \\(FN\\) as the number of \\(False Negatives\\), we can look at the key evaluation metrics in the following way:
# MAGIC 
# MAGIC \\[Accuracy =  \dfrac{TP +TN}{TP + FP + TN + FN}\\]
# MAGIC 
# MAGIC \\[Precision =  \dfrac{TP}{TP + FP}\\]
# MAGIC 
# MAGIC \\[Recall =  \dfrac{TP}{TP + FN}\\]
# MAGIC 
# MAGIC Accuracy is the simplest way to evaluate a classification model and does a good job of providing an initial sense of the quality of the model.  Yet, it is a relatively generic metric that does not help tune the model for specific business requirements.
# MAGIC 
# MAGIC Our goal with this project is to introduce a tool that would help improve the current state of the airline industry by providing customers (initially focusing on airlines) with a predicted delay for a flight that is not scheduled to depart for another 2 hours. With this context, we believe that a false negative (i.e. we predict no-delay, but the flight gets delayed) would be undesirable, but we also acknowledge that it would not be worse than the status quo: it would cause disappointment for customers, but no further damage. On the other hand, a false positive (i.e. we predict a delay and the flight departs on-time) might cause passengers and airlines to unnecessarily replan around the wrongly predicted flight delay; in practical terms, this may cause the airline to scramble to find an alternative aircraft to complete the flight or passengers to choose alternative ways of transportation that were less convenient to begin with.
# MAGIC 
# MAGIC Given the above considerations, but still wanting to achieve a balance between \\(Precision\\) and \\(Recall\\), we have chosen to use the \\(F_{\beta}\\) metric to measure the performance of our model:
# MAGIC 
# MAGIC \\[F_\beta=   \dfrac{(1 + \beta^2) \cdot (Precision \cdot Recall) }{ (\beta^2 \cdot Precision + Recall)}\\]  
# MAGIC 
# MAGIC Consistently with the business implications discussed above, we have chosen a \\(\beta = 0.5\\), which balances the \\(Precision\\) and \\(Recall\\) factors, while giving twice as much weight to \\(Precision\\) than \\(Recall\\). Our choice of metric was also influenced by the inherent imbalance in the datase - only about 20% of flights are delayed.

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Exploratory Data Analysis and Discussion of Challenges
# MAGIC 
# MAGIC We conducted extensive exploratory data analyses (EDA) on our dataset, both before and after the join. We limited our EDA primarily to flights from 2015 through 2018 since this subset is what we used for training our models. Below, we present a selection of our EDA below on the joined dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Summary

# COMMAND ----------

# MAGIC %md
# MAGIC Looking at our final join we ended up with 171 columns and three primary datastes:
# MAGIC - Training set (2015 - 2018)
# MAGIC - Test set (2019)
# MAGIC - COVID-19 set (2020 - 2021)
# MAGIC 
# MAGIC Below you can see the break out of the number of samples for each set.
# MAGIC 
# MAGIC Dataset | Samples
# MAGIC ------|-----------
# MAGIC Training | 24,236,876
# MAGIC Test | 7,393,362
# MAGIC COVID-19 | 10,641,410
# MAGIC 
# MAGIC By pulling in the 2020 and 2021 data we were able to get an additional 10.6 million samples. We will utilize this extra data to understand how the pandemic impacted flight delays and the F1-beta score of our models.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 3.2 Flights
# MAGIC 
# MAGIC The Flights portion of our data contains information about each flight recorded in the U.S. Department of Transportation's database [[2]](https://www.transtats.bts.gov/homepage.asp). This data can be broadly grouped into eleven categories:
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
# MAGIC From the set of features, we will be using `DEP_DEL15` as our outcome variable. This feature is an indicator that equals 1 when a flight has a departure delay greater than 15 minutes.
# MAGIC 
# MAGIC We were able to determine the meaning of each feature and decide each feature's relevance for our analysis and for developing a predictive model. For instance,  features in the "Cause of Delay" category should not be included in our model since these causes are determined after a delay is already known. However, we can analyze the frequency of delay causes to help us determine which other features should be prioritized for our model.
# MAGIC 
# MAGIC We focused on our target variable first to have a better understand of its distribution and how it varied between each dataset. We then began looking at different features that could impact our outcome variable. 
# MAGIC 
# MAGIC An interesting finding from our EDA is that each flight is duplicated in our dataset. Because of this, we will need to remove duplicates from our final, clean dataset. The original size of the entire dataset (not limited to ORD/ATL in Q1 2015) was 62,513,788, and once the duplicates are removed, this matches the number of flights we would expect.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2.1 Percent Flight Delays (2015 - 2018)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The bar chart below displays the frequency of our outcome variable in the training set. About 80% of flights are not delayed by an amount greater than 15 minutes, about 18% of flights are delayed, and 1.5% of the data is missing. This information is helpful since it shows our target classes are unbalanced, which will play a key role in how we develop our model and interpret the results. For our goal of performing a multinomial classification, we also analyzed the degree to which flights are delayed. There is a large skew in the data--most departing flights are either not delayed or leave early, and there is a long right tail as delay time increases.
# MAGIC 
# MAGIC ![Percent Delay](files/shared_uploads/ram.senth@berkeley.edu/eda/percent_flight_delay_low.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2.2 Percent Flight Delays (Training and Testing)

# COMMAND ----------

# MAGIC %md
# MAGIC The delay versus non-delay imbalance was consistent across years, except for  2020 which showed less frequent delays and more cancellations, especially in the months of March, April and May.  
# MAGIC Yet, as we are not predicting cancellations, our intuition is that our model will deliver consistent results for 2020 as long as the causes of delay have not significantly changed, which we plan to validate.
# MAGIC 
# MAGIC ![Percent Delay Train and Test](files/shared_uploads/ram.senth@berkeley.edu/eda/percent_flight_Delay_each_data_low.png)
# MAGIC 
# MAGIC The 2020 flight cancelations happend mostly in the months of March through May when the pandemic and lockdowns began. 
# MAGIC 
# MAGIC ![Percent cancelation 2020 Month](files/shared_uploads/ram.senth@berkeley.edu/eda/percent_flight_Delay_2020_month_low.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2.3 Percent Flight Delays vs Hour of Day

# COMMAND ----------

# MAGIC %md
# MAGIC We also suspected that earlier flights might be less delayed than later flights. The data suggests this is true.  
# MAGIC 
# MAGIC ![Percent Delay Hour](files/shared_uploads/ram.senth@berkeley.edu/eda/percent_flight_hour_low.png)
# MAGIC 
# MAGIC The percentage of delayed flights increases throughout the day from 0500 to 2000 hours. This leads us to belive that flight delays can be caused by the previous flight being delayed. In order to capture this phenomenon in our model we will need to create an engineered feature that flags a previous flight being delayed 2 hours before take off. We additionally analyzed the relationship between flight delays and day of week, month of year, quarter (which can be interpreted as a proxy for season), and airline. All of these qualitative variables show variation in their correlation to flight delay and will be worthwhile including in our initial prediction model. This will be covered in Section 4 (Feature Engineering).

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 3.3 Weather Stations

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The primary purpose of the weather stations dataset is to help us connect the origin and destination airports for each flight to the correct weather stations. This dataset typically associates weather stations with their closest airport and subsequently identifies airports using their ICAO codes. The flights dataset uses the IATA codes standard, so we used an intermediate dataset to convert from ICAO codes to IATA codes. Once we had the IATA codes for the airports in the weather stations dataset, we found that there were eight airports missing from the weather stations dataset: PSE, PPG, OGS, SPN, SJU, TKI, GUM, and XWA. Of these, SJU and GUM are large airports. Because these airports are missing from the dataset, **we will have to associate weather for flights from these airports using other closeby weather stations. More details on this are included later in the "Data Preparation" section.**

# COMMAND ----------

# MAGIC 
# MAGIC %md
# MAGIC 
# MAGIC ## 3.4 Weather

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The full weather dataset contains over 71 million rows and 123 features. We were able to narrow down this dataset to the correct locations by matching the station IDs from the weather stations dataset with the station IDs in the weather dataset. We additionally limited the features to those that seemed most relevant for this analysis, including wind (speed, direction), temperature, dew point, horizontal visibility, sea-level pressure, precipitation, and miscellaneous daily weather conditions (e.g. fog, thunder, tornado, and snow). This features can be categorized several  many categories.
# MAGIC 
# MAGIC 1. Origin Airport Continuous Data
# MAGIC 2. Destination Airport Continuous Data
# MAGIC 3. Origin Airport Binary Data
# MAGIC 2. Destination Airport Binary Data
# MAGIC 
# MAGIC We began or analysis by looking at the distirbutions of each weather feature. This allowed us to determine how these features needed to be normalized as well as identify the outliers that needed to be cleaned up. Next we began looking correlations between the continuous and binary weather with the outcome variable (DEP_DEL15).

# COMMAND ----------

# DBTITLE 1,Weather Table (2015-2021)
# Import the weather data, only showing the stations that we care about
df_weather = spark.read.parquet(WEATHER_LOC + '/clean_weather_data.parquet')
df_weather.createOrReplaceTempView("weather_vw")
display(df_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4.1 Weather Data Distributions

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC One example of the EDA we conducted on our weather features includes Wind Speed. Wind Speed is recorded in miles per hour. When we plot the distribution of wind speed, we find that the distribution has an extremely long right tail. This leads us to believe that some of the data is inaccurate. In addition to addressing missing data (null values), we will also need to address erroneous data like this. When we eliminate the incorrect data from our analysis, the distribution looks much better but still with a right skew. Most of the recorded wind is less than 20 mph.

# COMMAND ----------

# Get random sample of weather data to visualize
sampled_weather = df_weather.sample(fraction = 0.025)
sampled_weather_pd = sampled_weather.toPandas()

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize = (15, 5))
fig.suptitle('Average Wind Speed (mph)', fontsize = 16);
axes[0].set_title('Average Wind Speed < 20 mph');
sns.kdeplot(ax = axes[0], x = sampled_weather_pd[sampled_weather_pd['Avg_HourlyWindSpeed'] < 20]['Avg_HourlyWindSpeed']);
sns.kdeplot(ax = axes[1], x = sampled_weather_pd[sampled_weather_pd['Avg_HourlyWindSpeed'] < 50]['Avg_HourlyWindSpeed']);
sns.kdeplot(ax = axes[2], x = sampled_weather_pd['Avg_HourlyWindSpeed']);
axes[1].set_title('Average Wind Speed < 50 mph');
axes[2].set_title('Total distribution shows erroneous data');

# COMMAND ----------

# MAGIC %md
# MAGIC We can see a similar distribution for precipitation. This distribution has a significant right skew. The most frequently occurring value is 0, as shown in the far right figure since about 90% of entries indicate no rain. For nonzero precipitation, most of the hourly precipitation values fall below 0.1 inches. There is an increase in frequency around rounded values of hourly precipitation (ex. 0.1, 0.2, 0.3, etc.). This makes sense becuase our data only has precision to a hundredth of an inch.

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize = (15, 5))
fig.suptitle('Hourly Precipitation', fontsize = 16);
axes[0].set_title('Hourly Non-Zero Precipitation < 0.05');
sns.kdeplot(ax = axes[0], x = sampled_weather_pd[(sampled_weather_pd['Avg_Precip_Double'] > 0.00) &
                                  (sampled_weather_pd['Avg_Precip_Double'] < 0.05)]['Avg_Precip_Double']);
sns.kdeplot(ax = axes[1], x = sampled_weather_pd[(sampled_weather_pd['Avg_Precip_Double'] > 0.00) &
                                  (sampled_weather_pd['Avg_Precip_Double'] < 1)]['Avg_Precip_Double']);
plt.bar(x = sampled_weather_pd['NonZero_Rain'].value_counts().index, height = sampled_weather_pd['NonZero_Rain'].value_counts()/sum(sampled_weather_pd['NonZero_Rain'].value_counts()));
axes[2].set_xticks([0, 1]);
axes[1].set_title('Hourly Non-Zero Precipitation < 1');
axes[2].set_title('Distribution of Precipitation Frequency');
axes[2].set_xlabel('No Precipitation           Precipitation');

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4.2 Weather Data Correlations

# COMMAND ----------

# MAGIC %md
# MAGIC With the joined flights and weather datasets we began looking at the correlations between the weather data and the delay features.
# MAGIC 
# MAGIC ![Origin Weather Corr](files/shared_uploads/ram.senth@berkeley.edu/eda/weather_origin_corr_low.png)
# MAGIC 
# MAGIC - There is no direct correlation between the continuous origin weather data and aircraft delay
# MAGIC - There are strong correlations between these weather features and each other
# MAGIC   - Example: Elevation and Pressure
# MAGIC   - We can drop on of these features for each pair because they arent adding any new information
# MAGIC   
# MAGIC ![Dest Weather Corr](files/shared_uploads/ram.senth@berkeley.edu/eda/weather_dest_corr_low.png)
# MAGIC 
# MAGIC - There is no direct correlation between the continuous destination weather data and aircraft delay
# MAGIC - There are strong correlations between these weather features and each other
# MAGIC   - Example: Elevation and Pressure
# MAGIC   - We can drop on of these features for each pair because they arent adding any new information

# COMMAND ----------

# MAGIC %md
# MAGIC ![Hail Corr](files/shared_uploads/ram.senth@berkeley.edu/eda/hail_low.png)
# MAGIC 
# MAGIC - Looking at the binary weather features at the origin airport you can see a much stronger correlation with delay time
# MAGIC - There are stronger correlations with more extreme weather such as hail and snow and weaker correlations with just rain being present

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
# MAGIC We had to conduct significant data cleaning on the weather dataset in order to get it in a usable form. As mentioned earlier, we reduced the size of the dataset by filtering only to the weather stations either at airports or closeby to airports for the airports without a weather station on site (see more information about this below and in the Join Strategy section). 
# MAGIC 
# MAGIC Then, we identified and selected relevant variables beyond the station identifiers, which included wind, humidity, horizontal visibility, temperature, pressure change, dew point, precipitation, and present weather conditions. Some of these features, including present weather conditions and hourly sky conditions, contained subfeatures, which were split from the original data and converted into the correct format. Since all data was loaded in as strings, we converted the numerical values to floats. 
# MAGIC 
# MAGIC Our first pass at addressing missing weather information was to determine if there was any meaning behind the null values. For example, missing rain information likely indicates that there was no rain for that observation time and location. We filled in these values with zeros.
# MAGIC 
# MAGIC Finally, we reduce the size of our dataset by aggregating weather observations in the same hour. The final weather dataset, cleaned has 40 features and over 41 million rows.

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
# MAGIC In this project, we are dealing with time-series data, which require extra care in defining train and test splits.  More specifically, we are considering using Cross Validation as part of the tuning process of the model hyperparameters.  The main challenge with time-series data is avoiding data leakage, which can be caused any time data from the future is used to make a prediction about the present.  When it comes to Cross Validation, this means that we can't randomly split the the data across multiple sets, but we have to do it in such a way that preserves the sequential order of entries.  More specifically, this means that we have to always test our models with validation and test sets that contain later date flights than those used in the training phase.
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
# MAGIC # 4. Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 New Features
# MAGIC 
# MAGIC ### Holiday feature
# MAGIC Based on our research on and experiences with flight delays, we expected several additional factors to impact whether or not a flight would be delayed. First, we reasoned that flights around holidays could be more susceptible to delays. Because of this, we added a new feature that indicates 1 if a flight occurs on a holiday or within five days of a holiday, a "holiday window." However, ultimately because of collinearity between these two features, we only kept the holiday window feature.
# MAGIC 
# MAGIC ###Airline delay feature
# MAGIC Second, we created a categorical feature that groups airlines based on how frequently they experience flight delays. Based on our EDA, we created three categories: infrequent delays (airlines whose proportion of delayed flights is less than 17%), moderately frequent delays (proportion of delayed flights between 17% and 20%), and frequent delays (proportion of delayed flights greater than 20%). There are six airlines that fall into the infrequent category, seven airlines in the moderately frequent category, and six airlines in the frequent category. If our testing data sees a new airline, we default this feature to the moderately frequent category.
# MAGIC 
# MAGIC ![Airline Delay](files/shared_uploads/ram.senth@berkeley.edu/eda/perc_airline_delay_low.png)
# MAGIC 
# MAGIC ### Prior flight delay feature
# MAGIC Third, we added a binary feature that indicates whether or not the previous flight was delayed. We expect this feature to be significant since delays frequently propagate until the end of the day. We calculated this feature by sorting flights by departure date/time for each tail number and pulling the delay indicator for the previous flight *as long as* the previous flight was scheduled at least 2.25 hours *before* the current flight. We used this time constraint to prevent data leakage. The indicator is set to 0 if there are no previous flights on the same day. The table below shows some summary data utilizing this feature: 10% of aircraft that aren't delayed will be delayed on the next departure, and 90% will be on time. This insight suggests that this feature should be very helpful in our model. 
# MAGIC 
# MAGIC 
# MAGIC DEP_DEL15 | PREV_FLIGHT_DELAY | percentage
# MAGIC ------|--------|-----
# MAGIC 1 | 0 | 10%
# MAGIC 0 | 0 | 90%

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Data Transformations
# MAGIC With the clean data and these new features, we applied additional transformations to prepare our dataset for modeling. For example, we applied one-hot encoding on categorical features. Several time-based features were one-hot encoded, including quarter, month, and departure hour. For our quantitative features, we used a min-max scaler to ensure that our features were on the same scale. For  each of these, we ensured that the transformations were created based on the training set and then applied separately to the testing set. This prevented data leakage between our training and testing sets.
# MAGIC 
# MAGIC We removed cancelled flights and dropped columns with excessive amounts of missing values, including features based on flight diversions. We additionally dropped rows with missing values. About 10% of rows were dropped from our dataset.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 4.3 Final Feature List
# MAGIC We used the following features in our model to predict flight delays, which have been grouped into broad categories:
# MAGIC 
# MAGIC **Flight Characteristics**
# MAGIC 1. Local departure hour (one-hot encoded)
# MAGIC 2. Local day of week (one-hot encoded)
# MAGIC 3. Holiday window (within 5 days of a holiday) (Binary)
# MAGIC 4. Quarter of year (one-hot encoded)
# MAGIC 5. Month of year (one-hot encoded)
# MAGIC 6. Prior flight delayed indicator (Binary)
# MAGIC 7. Distance of flight (categorized into groups for each 250 miles) (one-hot encoded)
# MAGIC 
# MAGIC **Airport & Airline Characteristics**
# MAGIC 1. Origin and destination airport elevation (min-max scaler)
# MAGIC 2. Origin and destination airport size (small, medium, large) (one-hot encoded)
# MAGIC 3. Airline delay frequency (low, medium, high) (one-hot encoded)
# MAGIC 
# MAGIC **Hourly Weather Characteristics**
# MAGIC 1. Origin and destination temperature (min-max scaler)
# MAGIC 2. Origin and destination wind speed (min-max scaler)
# MAGIC 3. Origin and destination indicator for rain (binary)
# MAGIC 4. Origin and destination indicator for increasing pressure (binary)
# MAGIC 5. Origin and destination indicator for storm (binary)
# MAGIC 6. Origin and destination indicator for hail (binary)
# MAGIC 
# MAGIC Through our EDA, we discovered that many of features in our total feature set were correlated, so we excluded some correlated features from our models, including wet bulb temperature and pressure change. Finally, we excluded other features that we felt would likely not significantly impact model performance, including altimeter setting, sea level pressure, and wind direction. For our baseline model, we used a limited set of the features above and had a total of 106 features. After, we expanded to the full feature set above and used a total of 79 features.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4 Addressing Imbalance
# MAGIC 
# MAGIC As already stated, our data has a severe imbalance between our two classes. Because of this, we experimented with several different options of addressing this imbalance to add more emphasis to the minority class (delayed flights). We compared three primary strategies: class weights, oversampling the minority class, and undersampling the majority class. We ran some experimental models using each of the three methods and found that class weights consistently performed the worst and had the lowest \\(F_\beta\\) score. Oversampling the minority class and undersampling the majority class behaved very similarly, and we further adjusted the ratio by which we under or oversampled. We found that using a ratio of 3 worked well consistently across model types (Logistic Regression and Random Forests). This means that we only kept one-third of the majority class when we undersampled, or we tripled the size of the majority class when we oversampled. Although oversampling and undersampling had comparable performance, we ultimately decided to undersample the majority class. The primary driver for this decision was based on run time. We figured that adding even more data to our already large dataset would increase the time to fit and evaluate models. Because of the time constraint, we undersampled the majority class and have identified the models below for which undersampling was applied.

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Algorithm Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 5.1 Toy Example
# MAGIC 
# MAGIC We tested a toy example of our model by implementing logisitic regression from scratch. While we initially wanted to use our \\(F_\beta\\) metric as our loss function, we found that F scores are not differentiable, and we could therefore not calculate the gradient to use in gradient descent. As a result, we used accuracy for the loss function, which is differentiable, and additionally evaluated the model on the \\(F_\beta\\) score.
# MAGIC 
# MAGIC Logistic regression predictions are made using the sigmoid function. The output is a probability value between 0 and 1 and is calculated as:
# MAGIC 
# MAGIC \\[h_ \theta (x) =  \frac{\mathrm{1} }{\mathrm{1} + e^- \theta^Tx }\\]
# MAGIC 
# MAGIC The cost function using accuracy is defined as:
# MAGIC 
# MAGIC \\[J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}y^{i}\log(h_\theta(x^{i}))+(1-y^{i})\log(1-h_\theta(x^{i}))\\]
# MAGIC 
# MAGIC We can use the derivative of our cost function in a gradient descent algorithm, and the following equation defines how we update our model weights using this derivative:
# MAGIC 
# MAGIC \\[\theta_{j} = \theta_{j} - \alpha \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{i})-y^i)x_j^i\\]
# MAGIC 
# MAGIC We calculate our \\(F_\beta\\) score separate from this gradient descent algorithm and track these scores for each iteration of gradient descent.
# MAGIC 
# MAGIC 
# MAGIC ![Model From Scratch](files/shared_uploads/ram.senth@berkeley.edu/model_plots/lr_manual_accuracy_f_beta_low.png)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We used a subset of our data and features to train the model using this mathematical implementation of logistic regression. Our features included the presence of hail at the origin and destination airports, the presence of a storm at the origin and destination airports, prior flight delay, and holiday window. We found that our gradient descent algorithm successfully optimized for our loss function, and the loss based on accuracy is decreasing. We fit this toy example using 15 iterations, which takes about 15 minutes to fit. After about 10 iterations, our test loss is actually less than our training loss, which implies this toy model is underfitting, and we address this in the next iteration of our modeling process. Additionally, our \\(F_\beta\\) scores remained relatively constant over the iterations. This will be something we keep in mind when creating other models, as we will ultimately select the model with the best \\(F_\beta\\) score. We attribute this to the fact that we're not fitting on the full dataset, and we haven't optimized the model yet. 
# MAGIC 
# MAGIC Since we were able to successfully implement logistic regression with graident descent on the toy dataset, we expanded to the full dataset and additional features and used PySpark's MLlib packages to help fit and evaluate our models. In these next iterations, we will see how the F-Beta score behaves on the full dataset.

# COMMAND ----------

# The toy example code - running custom Logistic Regression on a subset of data.

# To keep it simple, we will use a subset of features.
cols = ['DEP_DEL15', 'origin_weather_Present_Weather_Hail', 'origin_weather_Present_Weather_Storm',
        'dest_weather_Present_Weather_Hail', 'dest_weather_Present_Weather_Storm', 'prior_delayed', 'Holiday_5Day']

def load_data():
    """
        Load and prep both training and test datasets.
    """
    # Create Transformers
    # Custom sampler that down samples the majority class to address class imbalance.
    down_sampler = DownSampler(labelCol=configuration.orig_label_col, ratio=3.0)
    # Custom transformer that takes a smaller subset of samples to make training faster and also vectorizes the features.
    custom_transformer = CustomTransformer(percentage=0.0001, inputCols=cols)

    # Load training data.
    train = spark.read.parquet(configuration.TRANSFORMED_TRAINING_DATA)
    display(train.groupBy(configuration.orig_label_col).count())
    displayHTML("""Table 1: Original Training Dataset""")

    # Training data transformations: Downsample majority class to address imbalance, take a subset and vectorize features.
    train_vectorized = custom_transformer.transform(down_sampler.transform(train))
    train_vectorized.cache()
    display(train_vectorized.groupBy(configuration.orig_label_col).count())
    displayHTML("""Table 2: Transformed Training Dataset""")

    # Load training data.
    test = spark.read.parquet(configuration.TRANSFORMED_2019_DATA)
    display(test.groupBy(configuration.orig_label_col).count())
    displayHTML("""Table 3: Original Test Dataset</p>""")

    # Test data transformations: Transformations: take a subset and vectorize features.
    test_vectorized = custom_transformer.transform(test)
    display(test_vectorized.groupBy(configuration.orig_label_col).count())
    displayHTML("""Table 4: Transformed Test Dataset</p>""")
    test_vectorized.cache()
    return train_vectorized, test_vectorized

def get_confusion_matrix_vals(y_vals):
    """
        Calculate true positive, true negatives, false positives, and false negatives for a logistic regression model
        Input: (y_pred, y_true)
    """
    output = ['tp', 'tn', 'fp', 'fn']
    y_pred = y_vals[0]  # int(y_vals[0] > 0.5)
    y_true = y_vals[1]
    yield ('tp', y_true*y_pred)
    yield ('tn', (1-y_true)*(1-y_pred))
    yield ('fp', (1-y_true)*y_pred)
    yield ('fn', y_true*(1-y_pred))
    

def f_beta_loss(dataRDD, W):
    """
    Compute differentiable f_beta_loss.
    Args:
        dataRDD - each record is a tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    """
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1]))
    epsilon = 1e-5
    beta = 0.5
    
    # F_beta loss
    cm = augmentedData.map(lambda x: (1 / (1 + np.exp(-x[0].dot(W))), x[1])).flatMap(get_confusion_matrix_vals).reduceByKey(lambda x, y: x+y).cache()
    cm = dict(cm.collect())
    
    precision = cm['tp'] / (cm['tp'] + cm['fp'] + epsilon)
    recall = cm['tp'] / (cm['tp'] + cm['fn'] + epsilon)
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + epsilon)
    f_beta_loss = 1-f_beta

    # F_beta loss as Dice loss
#     loss = augmentedData.map(lambda x: (1 / (1 + np.exp(-x[0].dot(W))), x[1])).map(lambda x: 1 - ((1 + beta**2) * x[1] * x[0])/(beta**2 * x[1] + x[0])).mean()
    
    # accuracy
    loss = augmentedData.map(lambda x: (-x[1]*(np.log(1 / (1 + np.exp(-x[0].dot(W))) + epsilon))\
                            -((1-x[1])*(np.log(1 - (1 / (1 + np.exp(-x[0].dot(W)))) + epsilon))))).mean()

    return loss, f_beta_loss

def GDUpdate(dataRDD, W, learningRate = 0.1):
    """
    Perform one OLS gradient descent step/update.
    Args:
        dataRDD - records are tuples of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    Returns:
        new_model - (array) updated coefficients, bias at index 0
    """
    # add a bias 'feature' of 1 at index 0
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1])).cache()
    
    beta = 0.5
    
    # gradient for accuracy
    grad = augmentedData.map(lambda x: (1 / (1 + np.exp(-x[0].dot(W))) - x[1])*x[0]).mean()
    
    # gradient for Dice loss
#     grad = augmentedData.map(lambda x: ((-(1 + beta**2) * x[1]**2 * (1 / (1 + np.exp(-x[0].dot(W))))*(1 - (1 / (1 + np.exp(-x[0].dot(W)))))) /  ((beta**2)*(1 / (1 + np.exp(-x[0].dot(W)))) + x[1])**2)*x[0]).mean() 

    new_model = W - learningRate*grad

   
    return new_model

def GradientDescent(trainRDD, testRDD, wInit, nSteps = 20, learningRate = 0.1, verbose = False):
    """
    Perform nSteps iterations of OLS gradient descent and 
    track loss on a test and train set. Return lists of
    test/train loss and the models themselves.
    """
    # initialize lists to track model performance
    train_history, test_history, train_f_beta_history, test_f_beta_history, model_history = [], [], [], [], []
    
    # perform n updates & compute test and train loss after each
    model = wInit
    for idx in range(nSteps): 
        bModel = sc.broadcast(model)
        model = GDUpdate(trainRDD, bModel.value, learningRate)
        training_loss, training_f_beta_loss = f_beta_loss(trainRDD, model) 
        test_loss, test_f_beta_loss = f_beta_loss(testRDD, model)
        
        # keep track of test/train loss for plotting
        train_history.append(training_loss)
        train_f_beta_history.append(training_f_beta_loss)
        test_history.append(test_loss)
        test_f_beta_history.append(test_f_beta_loss)
        model_history.append(model)
        
        # console output if desired
        if verbose:
            print("----------")
            print(f"STEP: {idx+1}")
            print(f"training loss: {training_loss}")
            print(f"test loss: {test_loss}")
            print(f"training f_beta: {training_f_beta_loss}")
            print(f"test f_beta: {test_f_beta_loss}")
            print(f"Model: {[round(w,3) for w in model]}")
    return train_history, test_history, train_f_beta_history, test_f_beta_history, model_history

def plotErrorCurves(trainLoss, testLoss, trainFBeta, testFBeta, title = None):
    """
    Helper function for plotting.
    Args: trainLoss (list of errors) , testLoss (list of errors)
    """
    fig, ax = plt.subplots(1,1)
    x = list(range(len(trainLoss)))[1:]
    ax.plot(x, trainLoss[1:], 'k--', label='Training Loss (Accuracy)')
    ax.plot(x, testLoss[1:], 'r--', label='Test Loss (Accuracy)')
    ax.plot(x, trainFBeta[1:], 'b--', label='Training Loss (F-Beta)')
    ax.plot(x, testFBeta[1:], 'c--', label='Test Loss (F-Beta)')
    ax.legend(loc='right')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    fig.savefig(f'{configuration.MODEL_PLOTS_BASE_PATH}/lr_manual_accuracy_f_beta_low')
    if title:
        plt.title(title)
    plt.show()
    
def run_regression():
    train_vectorized, test_vectorized = load_data()
    
    # Initialize weights/baseline model.
    BASELINE = np.array([0.2] + [0]*(len(cols) - 1))

    train_history, test_history, train_FBeta, test_FBeta, model_history = GradientDescent(train_vectorized.rdd, \
                                                                                      test_vectorized.rdd, \
                                                                                      BASELINE, nSteps=15, \
                                                                                      learningRate=0.25, verbose=True)
    plotErrorCurves(train_history, test_history, train_FBeta, test_FBeta, title = 'Logistic Regression Gradient Descent')

run_regression()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 5.2 Baseline Model

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We define our baseline model as a uniform classifier that always predicts a single class. The evaluation metrics for these baseline models are summarized below.
# MAGIC 
# MAGIC **Baseline Metrics for Model that Predicts Majority Class (No Delay)**
# MAGIC 
# MAGIC **Weighted Metrics**
# MAGIC 
# MAGIC Metric       | Train (2015-2018) | Test (2019) | Test (2020) | Test (2021)
# MAGIC --------------------|-----------|-----------|-----------|-----------
# MAGIC Precision           |  0.67  |  0.66  |  0.83  |  0.68
# MAGIC Recall              |  0.82  |  0.81  |  0.91  |  0.83
# MAGIC F1 score            |  0.74  |  0.73  |  0.87  |  0.75
# MAGIC \\(F_{\beta=0.5}\\) |  0.70  |  0.69  |  0.84  |  0.71
# MAGIC accuracy            |  0.82  |  0.81  |  0.91  |  0.83
# MAGIC 
# MAGIC **Binary Metrics**
# MAGIC 
# MAGIC Metric       | Train (2015-2018) | Test (2019) | Test (2020) | Test (2021)
# MAGIC --------------------|-----------|-----------|-----------|-----------
# MAGIC Precision           |  0.00  |  0.00  |  0.00  |  0.00
# MAGIC Recall              |  0.00  |  0.00  |  0.00  |  0.00
# MAGIC F1 score            |  0.00  |  0.00  |  0.00  |  0.00
# MAGIC \\(F_{\beta=0.5}\\) |  0.00  |  0.00  |  0.00  |  0.00
# MAGIC accuracy            |  0.82  |  0.81  |  0.91  |  0.83
# MAGIC 
# MAGIC 
# MAGIC **Baseline Metrics for Model that Predicts Minority Class (Delay)**
# MAGIC 
# MAGIC **Weighted Metrics**
# MAGIC 
# MAGIC Metric       | Train (2015-2018) | Test (2019) | Test (2020) | Test (2021)
# MAGIC --------------------|-----------|-----------|-----------|-----------
# MAGIC Precision           |  0.03  |  0.03  |  0.01  |  0.03
# MAGIC Recall              |  0.18  |  0.19  |  0.09  |  0.17
# MAGIC F1 score            |  0.06  |  0.06  |  0.02  |  0.05
# MAGIC \\(F_{\beta=0.5}\\) |  0.04  |  0.04  |  0.01  |  0.04
# MAGIC accuracy            |  0.18  |  0.19  |  0.09  |  0.17
# MAGIC 
# MAGIC **Binary Metrics**
# MAGIC 
# MAGIC Metric       | Train (2015-2018) | Test (2019) | Test (2020) | Test (2021)
# MAGIC --------------------|-----------|-----------|-----------|-----------
# MAGIC Precision           |  0.18  |  0.19  |  0.09  |  0.17
# MAGIC Recall              |  1.00  |  1.00  |  1.00  |  1.00
# MAGIC F1 score            |  0.31  |  0.31  |  0.17  |  0.30
# MAGIC \\(F_{\beta=0.5}\\) |  0.22  |  0.22  |  0.11  |  0.21
# MAGIC accuracy            |  0.18  |  0.19  |  0.09  |  0.17

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 5.3 Basic Logistic Regression Model
# MAGIC 
# MAGIC We expected logistic regression to be relatively easy to implement, interpret and scale. We implemented the model with PySpark and a decision boundary of 0.5 to categorize the probabilities into discrete classes: delay versus no delay. We set **L1 regularization=0.05**, **L2 regularization=0.01** and **iterations=30** and undersampled the majority class to address the class imbalance. The model contained 79 features and was optimized using the area under the curve (AOC) metric and then evaluated using \\(F_{\beta=0.5}\\).  
# MAGIC 
# MAGIC The model was trained on 2015 to 2018 data with no cross-validation and took 5 minutes to run. We tested the model against 2019 to 2021 data.

# COMMAND ----------

# MAGIC %md ## 5.4 Optimized Logistic Regression Model
# MAGIC 
# MAGIC As next step we ran a grid search for optimizing L1, L2 regularization as well as the number of iterations. Again, we implemented the model with PySpark and a decision boundary of 0.5 to categorize the probabilities into discrete classes: delay versus no delay. We used undersampling of majority class to address the class imbalance. The model contained 79 features and was optimized using the area under the curve (AOC) metric and then evaluated using \\(F_{\beta=0.5}\\).  
# MAGIC 
# MAGIC The model was trained on 2015 to 2018 data with year block and took 45 minutes to train. We tested the model against 2019 to 2021 data and obtained the following results:
# MAGIC 
# MAGIC **Weighted Metrics**
# MAGIC 
# MAGIC Metric | Train (2015-2018) | Test (2019) | Test (2020) | Test (2021)
# MAGIC ------|-----------|-----------|-----------|-----------
# MAGIC Precision           | 0.80 | 0.80 | 0.87 | 0.81
# MAGIC Recall              | 0.84 | 0.82 | 0.87 | 0.83
# MAGIC F1 score            | 0.82 | 0.80 | 0.87 | 0.82
# MAGIC \\(F_{\beta=0.5}\\) | 0.81 | 0.80 | 0.87 | 0.81
# MAGIC accuracy            | 0.82 | 0.82 | 0.88 | 0.83
# MAGIC 
# MAGIC **Binary Metrics**
# MAGIC 
# MAGIC Metric | Train (2015-2018) | Test (2019) | Test (2020) | Test (2021)
# MAGIC -------------------|---------|---------|---------|---------
# MAGIC Precision            | 0.51 | 0.52 | 0.30 | 0.51
# MAGIC Recall               | 0.38 | 0.39 | 0.30 | 0.38
# MAGIC F1 score             | 0.44 | 0.44 | 0.30 | 0.48
# MAGIC \\(F_{\beta=0.5}\\)  | 0.48 | 0.49 | 0.30 | 0.48
# MAGIC accuracy             | 0.82 | 0.82 | 0.87 | 0.83

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.5 Basic Random Forest Classification Model
# MAGIC 
# MAGIC We also considered random forest given its ensemble method of learning and emphasis on feature selection. We implemented the model with PySpark and categorized flights into discrete classes: delay versus no delay. Since random forests can sometimes overfit, we set n_estimators, the number of trees, and max_depth, the maximum depth of a tree. Having a large number of trees can reduce overfitting but can also take longer to run, and the deeper a tree, the more it risks overfitting. We set **n_estimators=30** and **max_depth=20** and undersampled the majority class to address the class imbalance. The model contained 79 features and was evaluated using the \\(F_{\beta=0.5}\\) metric. 
# MAGIC 
# MAGIC The model was trained on 2015 to 2018 data with no cross-validation and took 47 minutes to run. We tested the model against 2019 to 2021 data and obtained the following results:
# MAGIC 
# MAGIC **Weighted Metrics**
# MAGIC 
# MAGIC Baseline Metric | Train (2015-2018) | Test (2019) | Test (2020) | Test (2021)
# MAGIC ------|-----------|-----------|-----------|-----------
# MAGIC Precision           | 0.81 | 0.80 | 0.87 | 0.81
# MAGIC Recall              | 0.86 | 0.85 | 0.88 | 0.86
# MAGIC F1 score            | 0.83 | 0.82 | 0.88 | 0.83
# MAGIC \\(F_{\beta=0.5}\\) | 0.82 | 0.81 | 0.88 | 0.82
# MAGIC accuracy            | 0.82 | 0.82 | 0.88 | 0.83
# MAGIC 
# MAGIC **Binary Metrics**
# MAGIC 
# MAGIC Metric | Train (2015-2018) | Test (2019) | Test (2020) | Test (2021)
# MAGIC -------------------|---------|---------|---------|---------
# MAGIC Precision            | 0.55 | 0.55 | 0.33 | 0.53
# MAGIC Recall               | 0.38 | 0.38 | 0.29 | 0.37
# MAGIC F1 score             | 0.45 | 0.45 | 0.31 | 0.43
# MAGIC \\(F_{\beta=0.5}\\)  | 0.50 | 0.50 | 0.32 | 0.49
# MAGIC accuracy             | 0.82 | 0.82 | 0.88 | 0.83
# MAGIC 
# MAGIC Interestingly, our basic Logistic Regression and Random Forest models both produced strong \\(F_{\beta=0.5}\\) scores of 0.8 on the blind 2019 test data. Surprisingly, the \\(F_{\beta=0.5}\\) score on the 2020 data was even higher at 0.87 for Random Forest. We suspect it could be due to the fact that there were relatively fewer delayed flights in 2020 versus 2019.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.6 XGBoost
# MAGIC We also explored XGBoost, an implementation of gradient boosted decision trees designed for speed and performance. Boosting algorithms like XGBoost first build a weak model, and then learn about feature importance and parameters to build a better model. Surprisingly, this model produced the lowest \\(F_{\beta=0.5}\\) score of 0.5. We suspect this might be due to the fact that XGBoost tends to perform poorly on imbalanced data and data with outliers. Considering that Random Forest and Logistic Regression produced significantly better results, we decided to not move forward with gradient boosted models.

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Algorithm Implementation

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6.1 Optimized Logistic Regression Models

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Our final model consisted of 79 features and Utilized grid search on L1 and L2 Regularization, and Max Iterations. The tuned hyperparameters and results can be seen below. 
# MAGIC 
# MAGIC - L1 Regularization Term: .01
# MAGIC - L2 Regularization Term: .01
# MAGIC - Maximum Iterations: 100
# MAGIC - CV Method: Block Years
# MAGIC - Number of features: 76
# MAGIC 
# MAGIC **Weighted Metrics**
# MAGIC 
# MAGIC Metric | Train (2015-2018) | Test (2019) | Test (2020) | Test (2021)
# MAGIC ------|-----------|-----------|-----------|-----------
# MAGIC Precision           | 0.80 | 0.80 | 0.87 | 0.81
# MAGIC Recall              | 0.84 | 0.82 | 0.87 | 0.83
# MAGIC F1 score            | 0.82 | 0.80 | 0.87 | 0.82
# MAGIC \\(F_{\beta=0.5}\\) | 0.81 | 0.80 | 0.87 | 0.81
# MAGIC accuracy            | 0.82 | 0.82 | 0.88 | 0.83
# MAGIC 
# MAGIC **Binary Metrics**
# MAGIC 
# MAGIC Metric | Train (2015-2018) | Test (2019) | Test (2020) | Test (2021)
# MAGIC -------------------|---------|---------|---------|---------
# MAGIC Precision            | 0.51 | 0.52 | 0.30 | 0.51
# MAGIC Recall               | 0.38 | 0.39 | 0.30 | 0.38
# MAGIC F1 score             | 0.44 | 0.44 | 0.30 | 0.48
# MAGIC \\(F_{\beta=0.5}\\)  | 0.48 | 0.49 | 0.30 | 0.48
# MAGIC accuracy             | 0.82 | 0.82 | 0.87 | 0.83

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6.2 Error Analysis
# MAGIC 
# MAGIC 
# MAGIC We ran some error analysis on the test data from 2019 and 2020 and found that our errors seem to be generally correlated to lack of available flight data.  This can be seen by looking at the origin-destination airport pairs that top the list of worst performers from a \\(F_\beta\\) perspective, very much like the data related to origin and destination airports alone.  In terms of airlines, the airlines with the lowest values of \\(F_\beta\\) are very similar to those that showed the most delays during EDA. 
# MAGIC 
# MAGIC One interesting finding of the error analysis is the performance of our model in predicting the delay of a flight when the inbound aircraft is delayed ('prior_delayed' feature).  Our model produces better predictions and less false positives when there is no expected delay of the incoming flight.  This seems to indicate excessive aggressiveness by the model with this specific variable and could be corrected in future evolutions of the product.  This would be a very relevant area of focus because the 'prior_delay' feature is strongly correlated with the probability of delay of the departing flight.
# MAGIC 
# MAGIC When it comes to weather conditions, presence of rain seems to be one of the most influencial features, and our model seems to do a better job in using weather at the origin airport as a predictor of delay, while weather conditions and presence of rain at the destination airport are contributing less to the precision of the prediction.

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Conclusion

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 7.1 Key Takeaways
# MAGIC 
# MAGIC Flight delays can have significant economic consequences for airlines and passengers, which underscores the importance of being able to predict when a flight will be delayed. To tackle this challenge, we developed classification models that use flight and weather data from 2015 through 2021. Using 2015 through 2018 as our training set and 2019 through 2021 as our testing set, we developed a variety of models, evaluated their performance, and assessed the impact of the COVID-19 pandemic on flight delays. To choose the best model, we used the \\(F_{\beta=0.5}\\) score, calculated using the weighted average of \\(F_\beta\\) scores by class.
# MAGIC 
# MAGIC Ultimately our best model ended up being a cross-validated logisitic regression model with an \\(F_\beta\\) of 0.80 on our 2019 test set. This model performs significantly better than our baseline model of predicting only a single class, for which the \\(F_\beta\\) equals 0.11 when we predict only the minority class. Interestingly, our models performs even better when tested on the 2020 than either 2019 or 2021. We believe the primary driver for this is that there were overall fewer flight delays during 2020 than non-pandemic years. Our model performance is good considering the constraints of this project, but we have ideas for future work to improve our models even more.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1.1 Novelty
# MAGIC 
# MAGIC Our primary aspect of novelty in this project comes from our collection and use of 2020 and 2021 data. We were successfully able to ingest this data, which presented many challenges discussed further below, and we used this data to evaluate our models and the impact of the COVID-19 pandemic. 
# MAGIC 
# MAGIC Additionally, while we consider this to be a data science "best practice" rather than a new, novel solution, we took extra effort to ensure our data pipeline works efficiently and effectively for all types of data (ex. training vs testing, one year vs all years). This included creating custom classes for data preparation, cross valdiation, datasets, and model configuration and importing `.py` files into our modeling notebooks.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 7.2 Challenges
# MAGIC 
# MAGIC 1. *Downloading and Ingesting the 2020 & 2021 data* -- As mentioned earlier, downloading the 2020 and 2021 data reqiured switching data sources, which required significant time in terms of getting, ingesting, and deciphering the data. Importing and ingesting 450 GB of data from 2015-2021 was difficult, and we learned that limiting the amount of data earlier presented great advantages. By limiting the weather stations to those near airports, we reduced the ingestion time from 5 hours to 15 minutes. Additionally, by default, Spark created 1 MB parquet files, which were not efficient. We repartitioned the data to get ensure a more efficient file size. Finally, Spark did not infer the correct schema for our data, and we had to explicitly convert the data types.
# MAGIC 2. *Join Optimization* -- With such large datasets for weather and flights, we had to optimize how we joined the two tables. We grouped both flights and weather into one-hour windows and filtered the weather stations. The join ultimately ran in 25 minutes.
# MAGIC 3. *Data Leakage Issues* -- One key component of this analysis was ensuring that we only made predictions based on information known at least two hours before a flight's departure time. We had to iteratively update our join strategy and feature engineering processes to prevent data leakage. This presented some challenges for us and took some time, but ultimately, we believe we were successful in preventing all data leakage.
# MAGIC 4. *Data Standardization* -- Across our weather and flights datasets, identical data was frequently represented in different ways. For example, each dataset had its own time information. We had to standardize how this data was recorded by having available both UTC and local time data. Additionally, the weather stations dataset and the flights dataset contained information about airports using different types of codes. We had to translate between these codes using an intermediate dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 7.3 Limitations and Future Work
# MAGIC 
# MAGIC 1. *Underfit Models* -- Our \\(F_{\beta=0.5}\\) score is frequently lower for our test dataset than our training dataset. This implies that our model might be underfitting training data. While we were pretty happy with our \\(F_\beta\\) score results, there is more work to be done to optimize the bias-variance tradeoff for our model. Some of these ideas are discussed further below.
# MAGIC 2. *Tail Number Data* -- Airplanes are uniquely identified using tail numbers, and we were able to pull information about each tail number, including number of seats on the plane, number of engines, manufacturer, and manufactured date. This data comes from the Federal Aviation Administration and is available at https://www.faa.gov/licenses_certificates/aircraft_certification/aircraft_registry/releasable_aircraft_download/.  We wanted to include this data in our analysis since we intuitively thought that there could be correlations between these features and flight delay. For instance, it might make sense for planes with an older manufactured date to break down more often and therefore experience delays more frequently. Furthermore, we might think that larger planes (i.e. those with more seats more more engines) are prioritized over smaller planes for ensuring on-time departure since flight delays for larger planes will impact more people. However, we did not end up including this dataset because of the high proportion of null values. Additionally some initial EDA on the data showed minimal correlation between these features and departure delay, and some features were correlated with existing features in our model. For example, number of seats was correlated with flight distance. Future work might include re-integrating this data by addressing the null values and refitting the models.
# MAGIC 3. *Historical Weather Forecasts* -- Our current methodology relies on existing weather conditions to predict flight delay. However, it might make more sense to use weather forecasts. For example, two hours before flight departure, we could use the weather forecast for the departure time as features in our model. Finding, downloading, and ingesting this data presented logistical challenges given the context of this project, so we decided to leave this idea for future work.
# MAGIC 4. *Filling in Null Values* -- To address null values in our selected columns, we ended up dropping rows with any null. This resulted in a loss of data of about 10%. In the future, we'd like to explore options of addressing null values besides dropping them, including filling in data with means or medians or calculating historical rolling averages for weather data. 
# MAGIC 5. *Additional work with XGBoost Models* -- Considering the time constraints of this project, we did not have time to address the issues we came across when fitting XGBoost decision tree models. In the future, we would like to further pursue this model type, including running cross validation and optimizing hyperparameters.
# MAGIC 6. *More Feature Engineering* -- In the process of cleaning the data and created models, we learned the significant role that our selected features played in our model performance. While we did significant work to refine our final feature set and create new features, there is still room for improvement. This includes applying feature transformations, such as quadratic terms or logarithmic terms. This also includes creating new features based on existing data, such as proportion of delayed flights seen earlier at an airport during a day.
# MAGIC 7. *Iterating on Models Based on Error Analysis* -- Now that we have conducted an analysis on our existing models and analyzed where we're going wrong, we'd like to integrate this analysis back into the modeling process. In future work, we would intentionally modify our models to make improvements upon our incorrect predictions.

# COMMAND ----------

# MAGIC %md
# MAGIC # 8. Application of Course Concepts

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 1. *MapReduce* -- Understanding the fundamentals of MapReduce allowed us to optimize our data pipeline for our large dataset. Utilizing our knowledge of where the bottle necks in processing occurs, we were able to organize our pipeline in a manner that reduced our join time down to less than thirty minutes.
# MAGIC 2. *Bias-Variance Tradeoff* -- The bias-variance tradeoff balances creating a model that is too simple (so it will have high bias but low variance and thus under-fit the data) versus creating a model that is too complex (which would cause the model to have low bias but high variance and thus over-fit the data). The tradeoff is finding a model that has a balance between the two. Utilizing this knowledge, we adjusted our model to try and balance these tradeoffs. While our model is still slightly underfit, we are still happy with our result.
# MAGIC 3. *Gradient Descent* -- Our primary model (Logistic Regression) utilizes gradient decent. We used our learnings on model iterations, learning rate, loss functions and regularization to optimize our model.
# MAGIC 4. *SQL* -- We used SQL to combine the flights table and the weather table as well as for adhoc data queries.
# MAGIC 5. *Graphs and Page Rank* -- While not used in our current models, our future work would include creating a feature that “Page Ranked” the airports. We would turn all of the flights and airports into a graph and page rank the airports by the number of flights that go through them. This feature could help in the future to improve our model.
# MAGIC 6. *PySpark* -- We utilized PySpark’s prebuilt models, LogisticRegression, RandomForestClassifier and XgboostClassifier to create our three different types of models. As well as it’s ParamGridBuilder to tune our hyper parameters.

# COMMAND ----------

# MAGIC %md
# MAGIC # 9. Bibliography and References
# MAGIC 
# MAGIC [1] A Report by the Joint Economic Committee, US Senate, “Your Flight has Been Delayed Again: Flight Delays Cost Passengers, Airlines, and the US Economy Billions”, 2008
# MAGIC https://www.jec.senate.gov/public/index.cfm/democrats/2008/5/your-flight-has-been-delayed-again_1539
# MAGIC 
# MAGIC [2] U.S. Department of Transportation Bureau of Transportation Statistics, "Transtats", 2022 https://www.transtats.bts.gov/homepage.asp
# MAGIC 
# MAGIC [3] National Oceanic and Atmospheric Admistration, "Local Climatological Data", 2022 https://www.ncei.noaa.gov/products/land-based-station/local-climatological-data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # 10. Code Organization
# MAGIC Our code is organized as two sets of notebooks. First set of notebooks, located within the `library` folder, contains code that creates python classes that the second set of notebooks depend on. Some of these are hosted on the team Azure Blobstore while the rest are hosted on the DataBricks File System. The `init_cluster` notebook has scripts for mounting the blobstore under `DBFS:/mnt` and adding the individual library files to the Spark context. It also sets up the necessary Spark configuration options, including auth for accessing the blob store.
# MAGIC 
# MAGIC 1. <a href="$./init_cluster">Initialize Cluster</a>
# MAGIC 2. <a href="$./library/configuration">Configuration</a>
# MAGIC 3. <a href="$./library/data_preparer">Data Preparer</a>
# MAGIC 4. <a href="$./library/training_sumarizer">Training Summarizer</a>
# MAGIC 
# MAGIC The second set of notebooks are for running actual tasks like data ingestion, EDA, model training etc.  Below are the list of notebooks: 
# MAGIC 
# MAGIC 1. Data ingestion
# MAGIC 
# MAGIC    - <a href="$./data_loading/Load New Flights Data">Load New Flights Data</a>
# MAGIC    - <a href="$./data_loading/Load New Weather Data">Load New Weather Data</a>
# MAGIC    - <a href="$./data_loading/Full Data Pipeline">Full Data Pipeline</a>
# MAGIC 1. EDA
# MAGIC    - <a href="$./eda/eda_flights">Exploratory Data Analysis Of Flights Data</a>
# MAGIC    - <a href="$./eda/eda_weather">Exploratory Data Analysis Of Weather Data</a>
# MAGIC    - <a href="$./eda/eda_joined_final">Exploratory Data Analysis on the Joined Dataset</a> 
# MAGIC 1. Staging data for model training
# MAGIC    - <a href="$./data_processing/data_processing">Data Processing</a>
# MAGIC 1. Model training
# MAGIC    - <a href="$./models/Logistic Regression from Scratch">Logistic Regression from Scratch</a>
# MAGIC    - <a href="$./models/Logistic Regression - Model Search with Cross Validation">Logistic Regression</a>
# MAGIC    - <a href="$./models/Logistic Regression - Random Forest - Model Search with Cross Validation">Random Forest</a>
# MAGIC    - <a href="$./models/XGBoost_cvModel">XGBoost</a>
# MAGIC 1. Error analysis

# COMMAND ----------


