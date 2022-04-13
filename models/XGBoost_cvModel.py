# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost - Model Search with Cross Validation
# MAGIC 
# MAGIC In this notebook we implement the search of an optimal XGBoost model, leveraging a customized verion of the CrossValidation algorithm to take into account the time series nature of our data. We also include custom FBeta validator for using FBeta score during training.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook setup

# COMMAND ----------

# general imports
import sys
import time
import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pyspark.sql.functions import col, isnan, when, count, lit, explode, array

# magic commands
# %matplotlib inline
%reload_ext autoreload
%autoreload 2

from configuration_v02 import Configuration
from data_preparer_v05 import DataPreparer, CVSplitter, CVStrategies
from training_summarizer_v05 import TrainingSummarizer
configuration = Configuration()
data_preparer = DataPreparer(configuration)
cvStrategies = CVStrategies()
training_summarizer = TrainingSummarizer(configuration)

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
from datetime import datetime
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from curve_metrics import CurveMetrics
from custom_cv import CustomCrossValidator
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score

# COMMAND ----------

# MAGIC %md ## Load Data

# COMMAND ----------

MODEL_NAME = 'xgboost_baseline'
dataset_to_use = configuration.FULL_DATASET

MODEL_NAME = 'xgboost_toy'
dataset_to_use = configuration.TOY_DATASET


# COMMAND ----------


# Load smaller dataset
training_withsplits, training, testing = data_preparer.load_and_prep_data(spark, \
                                                                          dataset_to_use, \
                                                                          cvStrategies.no_cv, \
                                                                          apply_class_weights=False)

# Full data
# training_withsplits, training, testing = data_preparer.load_and_prep_data(spark, \
#                                                                                configuration.FULL_DATASET, \
#                                                                                cvStrategies.no_cv, \
#                                                                                apply_class_weights=False)

# COMMAND ----------

# display(training.filter(col('CANCELLED') != 1).filter(col('DEP_DEL15').isNull()))
display(training.select(configuration.label_col).distinct())

# COMMAND ----------

display(training[configuration.feature_cols])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training And Evaluation

# COMMAND ----------

# # single node implementation from https://databricks.com/notebooks/xgboost/xgboost4j-spark-example.html
# import xgboost as xgb
# import mlflow.xgboost
# from sklearn.metrics import precision_score, fbeta_score
# spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

# xgtrain = xgb.DMatrix(training[feature_cols].toPandas(), training['label'].toPandas())

# with mlflow.start_run():
#     # Auto-log the model parameters with mlflow.xgboost.autolog

#     mlflow.xgboost.autolog()

#     param = {'max_depth': 2, 
#            'objective': 'multi:softmax', 
#            'num_class':3, 
#            'nthread':8}

#     bst = xgb.train(param, xgtrain, 10)

#     # Load testing data into DMatrix
#     dtest = xgb.DMatrix(testing_2019[feature_cols].toPandas())

#     # Predict testing data
#     ypred = bst.predict(dtest)

#     # Calculate accuracy score
#     f_score = fbeta_score(testing_2019["label"],ypred, average=None, beta=0.5)

#     # Log precision score as a metric
#     mlflow.log_metric("f_score", f_score)
  
#     print("XGBoost Model Fbeta(0.5) Score:",f_score)

# COMMAND ----------

# XGBoost for Pyspark
# Example from https://docs.databricks.com/applications/machine-learning/train-model/xgboost.html
from sparkdl.xgboost import XgboostRegressor, XgboostClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# Define a XgboostRegressor model that takes an input column "features" by default and learns to predict the labels in the "cnt" column.

Xgboost = XgboostClassifier(num_workers=4, labelCol=configuration.label_col, missing=0.0)

grid = ParamGridBuilder()\
  .addGrid(Xgboost.max_depth, [1, 3, 5, 7, 9])\
  .addGrid(Xgboost.n_estimators, [10, 50, 100, 150, 200])\
  .build()

# Declare the CrossValidator, which performs the model tuning.
#cv = CrossValidator(estimator=xgb_regressor, evaluator=evaluator, estimatorParamMaps=paramGrid)
    
cv = CustomCrossValidator(estimator=Xgboost, estimatorParamMaps=grid, evaluator=BinaryClassificationEvaluator(),
                          splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)

start_time = time.time()
cv_model = cv.fit(training_withsplits)
end_time = time.time()
train_time_seconds = end_time - start_time



# COMMAND ----------

# Save trained model
cv_model.write().overwrite().save(f'{configuration.TRAINED_MODEL_BASE_PATH}/{MODEL_NAME}')



# COMMAND ----------

# Analyze and save results.
model_summary = training_summarizer.get_summary(cv_model, 'XG Boost', MODEL_NAME, \
                dataset_to_use, train_time_seconds, 'Undersample Majority Class', \
                cvStrategies.no_cv, training, testing)

training_summarizer.save_model_summary(spark, sc, dbutils, model_summary, overwrite=False)


# COMMAND ----------

test_predictions = cv_model.bestModel.transform(testing)

# COMMAND ----------

display(test_predictions.groupby(configuration.label_col, 'prediction').count())

# COMMAND ----------


