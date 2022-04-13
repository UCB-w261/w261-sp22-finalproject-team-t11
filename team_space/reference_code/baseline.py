# Databricks notebook source
# MAGIC %md
# MAGIC # Baseline Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

from pyspark.sql.functions import col, isnan, when, count
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pyspark
import datetime as dt
import pandas as pd

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")


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

FINAL_JOINED_DATA_ALL = f"{blob_url}/staged/final_joined_all"


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Transform Data

# COMMAND ----------

def load_data():
    df = spark.read.parquet(FINAL_JOINED_DATA_ALL)
    print(f'Initial row count in full dataset: {df.count()}')
    return df
# df = df.where(df['CANCELLED'] != 1)
# df = df.dropna()

# COMMAND ----------

def transform(df):
    cols_to_keep = ['DEP_DEL15','origin_weather_Avg_Elevation', 'origin_weather_Avg_HourlyAltimeterSetting', 'origin_weather_Avg_HourlyDewPointTemperature', 'origin_weather_Avg_HourlyDryBulbTemperature',  'origin_weather_Avg_HourlyRelativeHumidity', 'origin_weather_Avg_HourlySeaLevelPressure', 'origin_weather_Avg_HourlyStationPressure', 'origin_weather_Avg_HourlyVisibility', 'origin_weather_Avg_HourlyWetBulbTemperature', 'origin_weather_Avg_HourlyWindDirection', 'origin_weather_Avg_HourlyWindSpeed', 'origin_weather_Avg_Precip_Double', 'dest_weather_Avg_Elevation', 'dest_weather_Avg_HourlyAltimeterSetting', 'dest_weather_Avg_HourlyDewPointTemperature', 'dest_weather_Avg_HourlyDryBulbTemperature', 'dest_weather_Avg_HourlyRelativeHumidity', 'dest_weather_Avg_HourlySeaLevelPressure', 'dest_weather_Avg_HourlyStationPressure', 'dest_weather_Avg_HourlyVisibility', 'dest_weather_Avg_HourlyWetBulbTemperature', 'dest_weather_Avg_HourlyWindDirection', 'dest_weather_Avg_HourlyWindSpeed', 'dest_weather_Avg_Precip_Double', 'YEAR']


    new_df = df.select(*cols_to_keep).dropna()
    print(f'Row counts after dropping NA: {new_df.count()}')
    return new_df

# COMMAND ----------

def summarize(df):
    df.select("DEP_DEL15").describe().show()
    print(f'df.filter(DEP_DEL15 == 0).count()={df.filter(col('DEP_DEL15') == 0).count()}')
    print(f'df.filter(DEP_DEL15 == 1).count()={df.filter(col('DEP_DEL15') == 1).count()}')


# COMMAND ----------

# Load the data
df = transform(load_data())
# split into test and train
df_train = df.where(df["YEAR"] < 2019)
df_test_2019 = df.where(df["YEAR"] == 2019)

# COMMAND ----------

summarize(df)

# COMMAND ----------

# MAGIC %md # Train and Test Model

# COMMAND ----------

def train_model(df_train, features, maxIter = 5):
    assembler = VectorAssembler(inputCols=features,outputCol="features")
    df_train_vector = assembler.transform(df_train)
    df_test_2019_vector = assembler.transform(df_test_2019)
    
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'DEP_DEL15', maxIter = maxIter)
    lrModel = lr.fit(df_train_vector)
    return lr, lrModel

def test_model(lrModel, df_test, features):
    assembler = VectorAssembler(inputCols=features,outputCol="features")
    df_test_vector = assembler.transform(df_test)
    return lrModel.transform(df_test_vector)


# COMMAND ----------

def train_and_test(df_train, df_test):
    features =  ['origin_weather_Avg_Elevation', 'origin_weather_Avg_HourlyAltimeterSetting', 'origin_weather_Avg_HourlyDewPointTemperature', 'origin_weather_Avg_HourlyDryBulbTemperature',  'origin_weather_Avg_HourlyRelativeHumidity', 'origin_weather_Avg_HourlySeaLevelPressure', 'origin_weather_Avg_HourlyStationPressure', 'origin_weather_Avg_HourlyVisibility', 'origin_weather_Avg_HourlyWetBulbTemperature', 'origin_weather_Avg_HourlyWindDirection', 'origin_weather_Avg_HourlyWindSpeed', 'origin_weather_Avg_Precip_Double', 'dest_weather_Avg_Elevation', 'dest_weather_Avg_HourlyAltimeterSetting', 'dest_weather_Avg_HourlyDewPointTemperature', 'dest_weather_Avg_HourlyDryBulbTemperature', 'dest_weather_Avg_HourlyRelativeHumidity', 'dest_weather_Avg_HourlySeaLevelPressure', 'dest_weather_Avg_HourlyStationPressure', 'dest_weather_Avg_HourlyVisibility', 'dest_weather_Avg_HourlyWetBulbTemperature', 'dest_weather_Avg_HourlyWindDirection', 'dest_weather_Avg_HourlyWindSpeed', 'dest_weather_Avg_Precip_Double', 'YEAR']

    lr, lrModel = train_model(df_train, features, maxIter=10)
    predictions = test_model(lrModel, df_test, features)
    predictions.cache()
    return lr, lrModel, predictions

# COMMAND ----------

lr, lrModel, predictions = train_and_test(df_train, df_test_2019)
# predictions.select('label', 'features', 'rawPrediction', 'prediction', 'probability').toPandas().head(5)

# COMMAND ----------

def summarize_model(lrModel, predictions):
    trainingSummary = lrModel.summary

    # Obtain the objective per iteration
#     objectiveHistory = trainingSummary.objectiveHistory
#     print("objectiveHistory:")
#     for objective in objectiveHistory:
#         print(objective)
        
    trainingSummary.roc.show()
    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))
    
#     prtest = predictions.pr.toPandas()
#     print(f'Test recall:{prtest["recall"]}')
#     print(f'Test precision:{prtest["precision"]}')

    # Set the model threshold to maximize F-Measure
#     fMeasure = trainingSummary.fMeasureByThreshold
#     maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
#     bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
#         .select('threshold').head()['threshold']
    
#     lr.setThreshold(bestThreshold)

#     # Get model F-Beta Score -- BINARY CLASSIFICATION METRICS DOESN'T HAVE AN F BETA CALCULATION?
#     beta = 0.5
#     multi_metrics = BinaryClassificationMetrics(predictions.select("DEP_DEL15", "prediction").rdd) # rdd should be predictions and labels
#     f_score = multi_metrics.fMeasure(label = 1, beta = beta) # i'm not sure what label is
#     print(f'F-Beta score: {f_score}')

# COMMAND ----------

summarize_model(lrModel, predictions)
# predictions.toPandas().head(5)

# COMMAND ----------

# predictions.toPandas().head(5)
display(predictions.where(col('DEP_DEL15') != col('prediction')))
# display(predictions.select("DEP_DEL15", "prediction"))


# COMMAND ----------


