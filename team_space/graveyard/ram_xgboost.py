# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost - Model Search with Cross Validation
# MAGIC 
# MAGIC In this notebook we implement the search of an optimal XGBoost model, leveraging a customized verion of the CrossValidation algorithm to take into account the time series nature of our data. We also include custom FBeta validator for using FBeta score during training.
# MAGIC 
# MAGIC XGBoost for Pyspark
# MAGIC 
# MAGIC Example from https://docs.databricks.com/applications/machine-learning/train-model/xgboost.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook setup

# COMMAND ----------

# general imports
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# import ast
# from pyspark.sql.functions import col, isnan, when, count, lit, explode, array

from sparkdl.xgboost import XgboostRegressor, XgboostClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from datetime import datetime
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# import pyspark.sql.functions as F
# from pyspark.sql.types import *

# magic commands
%matplotlib inline

from curve_metrics import CurveMetrics
from custom_cv import CustomCrossValidator
from configuration_v02 import Configuration
from data_preparer_v05 import DataPreparer, CVSplitter, CVStrategies
from training_summarizer_v05 import TrainingSummarizer
configuration = Configuration()
data_preparer = DataPreparer(configuration)
cvStrategies = CVStrategies()
training_summarizer = TrainingSummarizer(configuration)

# COMMAND ----------

def wrapper(dataset, model_name, params_maxDepth, params_estimators, evaluator, cv_strategy='None'):
    # Load data
    training_withsplits, training, testing = data_preparer.load_and_prep_data(spark, dataset, cv_strategy)

    class_imbalance_strategy = 'Undersample Majority Class'

    # Create classifier
    classifier = XgboostClassifier(num_workers=4, labelCol=configuration.label_col, missing=0.0)
    
    grid = ParamGridBuilder()\
      .addGrid(classifier.max_depth, params_maxDepth)\
      .addGrid(classifier.n_estimators, params_estimators)\
      .build()

    cv = CustomCrossValidator(estimator=classifier, estimatorParamMaps=grid, evaluator=evaluator,
                              splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)

#     start_time = time.time()
#     cv_model = cv.fit(training_withsplits)
#     end_time = time.time()
#     train_time_seconds = end_time - start_time

    # Save trained model
    cv_model.write().overwrite().save(f'{configuration.TRAINED_MODEL_BASE_PATH}/{model_name}')

    # Analyze and save results.
#     model_summary = training_summarizer.get_summary(cv_model, 'eXtreme Gradient Boosting', model_name, \
#                     dataset, train_time_seconds, class_imbalance_strategy, \
#                     cv_strategy, training, testing)

#     training_summarizer.save_model_summary(spark, sc, dbutils, model_summary, overwrite=False)
    return cvModel, training, testing
    


# COMMAND ----------

# MAGIC %md #Toy Dataset

# COMMAND ----------

cvModel_toy, training_toy, testing_toy = wrapper(configuration.TOY_DATASET, 'ram_xgb_toy', params_maxDepth=[6], \
                                                 params_estimators=[100], evaluator=BinaryClassificationEvaluator())

# COMMAND ----------

training_toy.groupby(configuration.label_col).count()

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

# Analyze and save results.


# COMMAND ----------

test_predictions = cv_model.bestModel.transform(testing)

# COMMAND ----------

display(test_predictions.groupby(configuration.label_col, 'prediction').count())

# COMMAND ----------


