# Databricks notebook source
# MAGIC %md
# MAGIC # Model Search with Cross Validation
# MAGIC 
# MAGIC In this notebook we implement the search of an optimal model leveraging a customized verion of the CrossValidation algorithm to take into account the time series nature of our data.

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

# training = training.filter(col('CANCELLED') == 0).na.drop(subset=['DEP_DEL15'])
# testing = testing.filter(col('CANCELLED') == 0).na.drop(subset=['DEP_DEL15'])

# COMMAND ----------

# Temporary fix for Holiday and Holiday_5Day to convert to integer
# Will remove after we run data processing next time

training = training.withColumn('Holiday', col('Holiday').cast('integer'))
training = training.withColumn('Holiday_5Day', col('Holiday_5Day').cast('integer'))
testing = testing.withColumn('Holiday', col('Holiday').cast('integer'))
testing = testing.withColumn('Holiday_5Day', col('Holiday_5Day').cast('integer'))

# COMMAND ----------

display(training.filter(col('CANCELLED') != 1).filter(col('DEP_DEL15').isNull()))
display(training.select('DEP_DEL15').distinct())

# COMMAND ----------

display(training)
display(testing)

# COMMAND ----------

# vectorize features
feature_cols = ['origin_weather_Present_Weather_Hail', 
                'origin_weather_Present_Weather_Storm', 
                'dest_weather_Present_Weather_Hail', 
                'dest_weather_Present_Weather_Storm', 
                'prior_delayed', 
                '_QUARTER', 
                '_MONTH', 
                '_DAY_OF_WEEK', 
                '_CRS_DEPT_HR', 
                '_DISTANCE_GROUP', 
                '_AIRLINE_DELAYS', 
                '_origin_airport_type', 
                '_dest_airport_type',
                'Holiday',
                'Holiday_5Day']

label_col = 'DEP_DEL15'

if 'features' in training.columns:
    training = training.drop('features')
training = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(training).withColumnRenamed(label_col, 'label').cache()

if 'features' in testing.columns:
    testing = testing.drop('features')
testing = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(testing).withColumnRenamed(label_col, 'label').cache()

# COMMAND ----------

display(training[feature_cols])

# COMMAND ----------

# MAGIC %md
# MAGIC # Logistic Regression

# COMMAND ----------

# import custom cv module
spark.sparkContext.addPyFile("dbfs:/custom_cv.py")
from custom_cv import CustomCrossValidator

spark.sparkContext.addPyFile("dbfs:/user/ram.senth@berkeley.edu/fbeta_evaluator.py")
from fbeta_evaluator import FBetaEvaluator

# set up grid search: estimator, set of params, and evaluator
lr = LogisticRegression(labelCol="label", featuresCol="features")
grid = ParamGridBuilder()\
            .baseOn({lr.labelCol: 'label'}) \
            .baseOn([lr.predictionCol, 'prediction']) \
            .addGrid(lr.regParam, [1.0, 2.0]) \
            .addGrid(lr.maxIter, [1, 5]) \
            .build()
evaluator = BinaryClassificationEvaluator()
evaluator = FBetaEvaluator()

# COMMAND ----------

# create dictionary of dataframes for custom cv fn to loop through
# assign train and test based on time series split 
# df1: Train: 2015 - Test: 2016
# df2: Train: 2015, 2016 - Test: 2017
# df3: Train: 2015, 2016, 2017 - Test: 2018

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

# COMMAND ----------

# run cross validation
cv = CustomCrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator,
     splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)

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

# MAGIC %md
# MAGIC ## Random Forest

# COMMAND ----------

# import custom cv module
spark.sparkContext.addPyFile("dbfs:/custom_cv.py")
from custom_cv import CustomCrossValidator

# set up grid search: estimator, set of params, and evaluator
rf = RandomForestClassifier(labelCol="label", featuresCol="features")
grid = ParamGridBuilder()\
            .addGrid(rf.maxDepth, [5, 10])\
            .addGrid(rf.numTrees, [10, 15])\
            .build()
evaluator = BinaryClassificationEvaluator()
evaluator = FBetaEvaluator()

# COMMAND ----------

# create dictionary of dataframes for custom cv fn to loop through
# assign train and test based on time series split 
# df1: Train: 2015 - Test: 2016
# df2: Train: 2015, 2016 - Test: 2017
# df3: Train: 2015, 2016, 2017 - Test: 2018

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

# COMMAND ----------

# run cross validation
cv = CustomCrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator,
     splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)

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

display(training.filter((training.FL_DATE >= '2015-04') &  (training.FL_DATE <= '2015-06')))

# COMMAND ----------

# create dictionary of dataframes for custom cv fn to loop through
# assign train and test based on time series split 
# df1: Train: 2015 - Test: 2016
# df2: Train: 2015, 2016 - Test: 2017
# df3: Train: 2015, 2016, 2017 - Test: 2018

d = {}

d['df1'] = training.filter(training.YEAR <= 2016)\
                   .withColumn('cv', F.when((training.YEAR == 2015), 'train')
                                         .otherwise('test'))

d['df2'] = training.filter(training.YEAR <= 2017)\
                   .withColumn('cv', F.when((training.YEAR <= 2016), 'train')
                                         .otherwise('test'))

d['df3'] = training.filter(training.YEAR <= 2018)\
                   .withColumn('cv', F.when((training.YEAR <= 2017), 'train')
                                         .otherwise('test'))
