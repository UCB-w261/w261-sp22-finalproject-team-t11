# Databricks notebook source
# MAGIC %md
# MAGIC # Model Search with Cross Validation
# MAGIC 
# MAGIC In this notebook we implement the search of an optimal model leveraging a customized verion of the CrossValidation algorithm to take into account the time series nature of our data.
# MAGIC 
# MAGIC ## Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC ### Notebook setup

# COMMAND ----------

# general imports
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pyspark.sql.functions import col, isnan, when, count

# magic commands
%matplotlib inline
%reload_ext autoreload
%autoreload 2

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

TRANSFORMED_TRAINING_DATA = f"{blob_url}/transformed/training"
TRANSFORMED_2019_DATA = f"{blob_url}/transformed/2019"


# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.ml.linalg import DenseVector, SparseVector, Vectors
from pyspark.ml.feature import VectorAssembler, StandardScaler

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# COMMAND ----------

# Load training data
training = spark.read.parquet(TRANSFORMED_TRAINING_DATA)
testing = spark.read.parquet(TRANSFORMED_2019_DATA)

# Take a subset of the data
# training = training_all.sample(fraction = 0.025)
# testing = testing_all.sample(fraction = 0.025)

# training = training.filter(col('CANCELLED') == 0).na.drop(subset=['DEP_DEL15'])
# testing = testing.filter(col('CANCELLED') == 0).na.drop(subset=['DEP_DEL15'])

# COMMAND ----------

display(training.filter(col('CANCELLED') != 1).filter(col('DEP_DEL15').isNull()))
display(training.select('DEP_DEL15').distinct())

# COMMAND ----------

display(training)
display(testing)

# COMMAND ----------

# vectorize features
feature_cols = ['origin_weather_Avg_Elevation',
                 'origin_weather_Avg_HourlyAltimeterSetting',
                 'origin_weather_Avg_HourlyDewPointTemperature',
                 'origin_weather_Avg_HourlyDryBulbTemperature',
                 'origin_weather_Avg_HourlyRelativeHumidity',
                 'origin_weather_Avg_HourlySeaLevelPressure',
                 'origin_weather_Avg_HourlyStationPressure',
                 'origin_weather_Avg_HourlyVisibility',
                 'origin_weather_Avg_HourlyWetBulbTemperature',
                 'origin_weather_Avg_HourlyWindDirection',
                 'origin_weather_Avg_HourlyWindSpeed',
                 'origin_weather_Avg_Precip_Double',
                 'origin_weather_Trace_Rain',
                 'origin_weather_NonZero_Rain',
                 'origin_weather_HourlyPressureTendency_Increasing',
                 'origin_weather_Calm_Winds',
                 'origin_weather_Sky_Conditions_CLR',
                 'origin_weather_Sky_Conditions_FEW',
                 'origin_weather_Sky_Conditions_SCT',
                 'origin_weather_Sky_Conditions_BKN',
                 'origin_weather_Sky_Conditions_OVC',
                 'origin_weather_Sky_Conditions_VV',
                 'origin_weather_Present_Weather_Drizzle',
                 'origin_weather_Present_Weather_Rain',
                 'origin_weather_Present_Weather_Snow',
                 'origin_weather_Present_Weather_SnowGrains',
                 'origin_weather_Present_Weather_IceCrystals',
                 'origin_weather_Present_Weather_Hail',
                 'origin_weather_Present_Weather_Mist',
                 'origin_weather_Present_Weather_Fog',
                 'origin_weather_Present_Weather_Smoke',
                 'origin_weather_Present_Weather_Dust',
                 'origin_weather_Present_Weather_Haze',
                 'origin_weather_Present_Weather_Storm',
                 'dest_weather_Avg_Elevation',
                 'dest_weather_Avg_HourlyAltimeterSetting',
                 'dest_weather_Avg_HourlyDewPointTemperature',
                 'dest_weather_Avg_HourlyDryBulbTemperature',
                 'dest_weather_Avg_HourlyRelativeHumidity',
                 'dest_weather_Avg_HourlySeaLevelPressure',
                 'dest_weather_Avg_HourlyStationPressure',
                 'dest_weather_Avg_HourlyVisibility',
                 'dest_weather_Avg_HourlyWetBulbTemperature',
                 'dest_weather_Avg_HourlyWindDirection',
                 'dest_weather_Avg_HourlyWindSpeed',
                 'dest_weather_Avg_Precip_Double',
                 'dest_weather_Trace_Rain',
                 'dest_weather_NonZero_Rain',
                 'dest_weather_HourlyPressureTendency_Increasing',
                 'dest_weather_Calm_Winds',
                 'dest_weather_Sky_Conditions_CLR',
                 'dest_weather_Sky_Conditions_FEW',
                 'dest_weather_Sky_Conditions_SCT',
                 'dest_weather_Sky_Conditions_BKN',
                 'dest_weather_Sky_Conditions_OVC',
                 'dest_weather_Sky_Conditions_VV',
                 'dest_weather_Present_Weather_Drizzle',
                 'dest_weather_Present_Weather_Rain',
                 'dest_weather_Present_Weather_Snow',
                 'dest_weather_Present_Weather_SnowGrains',
                 'dest_weather_Present_Weather_IceCrystals',
                 'dest_weather_Present_Weather_Hail',
                 'dest_weather_Present_Weather_Mist',
                 'dest_weather_Present_Weather_Fog',
                 'dest_weather_Present_Weather_Smoke',
                 'dest_weather_Present_Weather_Dust',
                 'dest_weather_Present_Weather_Haze',
                 'dest_weather_Present_Weather_Storm',
                 'prior_delayed',
                 'Holiday',
                 'Holiday_5Day',
                 '_QUARTER',
                 '_MONTH',
                 '_DAY_OF_WEEK',
                 '_CRS_DEPT_HR',
                 '_DISTANCE_GROUP',
                 '_AIRLINE_DELAYS',
                 '_origin_airport_type',
                 '_dest_airport_type']


# feature_cols = ['origin_weather_Present_Weather_Hail', 
#                 'origin_weather_Present_Weather_Storm', 
#                 'dest_weather_Present_Weather_Hail', 
#                 'dest_weather_Present_Weather_Storm', 
#                 'prior_delayed', 
#                 '_QUARTER', 
#                 '_MONTH', 
#                 '_DAY_OF_WEEK', 
#                 '_CRS_DEPT_HR', 
#                 '_DISTANCE_GROUP', 
#                 '_AIRLINE_DELAYS', 
#                 '_origin_airport_type', 
#                 '_dest_airport_type',
#                 'Holiday',
#                 'Holiday_5Day']


label_col = 'DEP_DEL15'

if 'features' in training.columns:
    training = training.drop('features')
training = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(training).withColumnRenamed(label_col, 'label').cache()
#training = training.withColumn("YEAR", col(training_all[YEAR]))

if 'features' in testing.columns:
    testing = testing.drop('features')
testing = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(testing).withColumnRenamed(label_col, 'label').cache()
#testing = testing[feature_cols]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cross Validation Options

# COMMAND ----------

def CV_rolling_window():
    d = {}

    d['df1'] = training.filter(training.YEAR <= 2016)\
                       .withColumn('cv', F.when(training.YEAR == 2015, 'train')
                                             .otherwise('test'))

    d['df2'] = training.filter(training.YEAR <= 2017)\
                       .withColumn('cv', F.when(training.YEAR <= 2016, 'train')
                                             .otherwise('test'))

    d['df3'] = training.filter(training.YEAR <= 2018)\
                       .withColumn('cv', F.when(training.YEAR <= 2017, 'train')
                                             .otherwise('test'))
    return d

def CV_block_by_quarter():
    d = {}
    
    d['df1'] = training.filter((training.YEAR.isin(2015,2016)) & (training.QUARTER == 1))\
                       .withColumn('cv', F.when((training.YEAR == 2015), 'train')
                                             .otherwise('test'))

    d['df2'] = training.filter((training.YEAR.isin(2015,2016)) & (training.QUARTER == 3))\
                       .withColumn('cv', F.when((training.YEAR == 2015), 'train')
                                             .otherwise('test'))

    d['df3'] = training.filter((training.YEAR.isin(2016,2017)) & (training.QUARTER == 2))\
                       .withColumn('cv', F.when((training.YEAR == 2016), 'train')
                                             .otherwise('test'))

    d['df4'] = training.filter((training.YEAR.isin(2016,2017)) & (training.QUARTER == 4))\
                       .withColumn('cv', F.when((training.YEAR == 2016), 'train')
                                             .otherwise('test'))

    d['df5'] = training.filter((training.YEAR.isin(2017,2018)) & (training.QUARTER == 1))\
                       .withColumn('cv', F.when((training.YEAR == 2017), 'train')
                                             .otherwise('test'))

    d['df6'] = training.filter((training.YEAR.isin(2017,2018)) & (training.QUARTER == 3))\
                       .withColumn('cv', F.when((training.YEAR == 2017), 'train')
                                             .otherwise('test'))

    d['df7'] = training.filter((training.YEAR.isin(2015,2018)) & (training.QUARTER == 2))\
                       .withColumn('cv', F.when((training.YEAR == 2015), 'train')
                                             .otherwise('test'))

    d['df8'] = training.filter((training.YEAR.isin(2015,2018)) & (training.QUARTER == 4))\
                       .withColumn('cv', F.when((training.YEAR == 2015), 'train')
                                             .otherwise('test'))
    return d

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest

# COMMAND ----------

# import custom cv module
spark.sparkContext.addPyFile("dbfs:/custom_cv.py")
from custom_cv import CustomCrossValidator
spark.sparkContext.addPyFile("dbfs:/user/ram.senth@berkeley.edu/fbeta_evaluator.py")
from fbeta_evaluator import FBetaEvaluator

# set up grid search: estimator, set of params, and evaluator
rf = RandomForestClassifier(labelCol="label", featuresCol="features")
grid = ParamGridBuilder()\
            .addGrid(rf.maxDepth, [5, 10])\
            .addGrid(rf.numTrees, [10, 15])\
            .build()
# evaluator = FBetaEvaluator()
evaluator = BinaryClassificationEvaluator()

# COMMAND ----------

# run cross validation
cv = CustomCrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator,
     splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
d = CV_rolling_window()
cvModel = cv.fit(d)

# COMMAND ----------

# make predictions
predictions = cvModel.transform(training)
display(predictions.groupby('label', 'prediction').count())

# COMMAND ----------

print (f'F1-score: {cvModel.avgMetrics[0]}')
print (f'Weighted Precision: {cvModel.avgMetrics[1]}')
print (f'Weighted Recall: {cvModel.avgMetrics[2]}')
print (f'Accuracy: {cvModel.avgMetrics[3]}')


# COMMAND ----------

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))

# COMMAND ----------

training_summary = cvModel.bestModel.summary
importance = ExtractFeatureImp(cvModel.bestModel.featureImportances, predictions, "features").head(25)
importance.sort_values(by=["score"], inplace = True, ascending=False)
display(importance)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(20,8))

fig1 = plt.barh(importance["name"], importance["score"])

plt.title("Feature Importance", fontsize = 14)
ax.invert_yaxis()
fig.show()

# COMMAND ----------


