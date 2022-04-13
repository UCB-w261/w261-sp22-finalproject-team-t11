# Databricks notebook source
# MAGIC %md
# MAGIC # Logistic Regression

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

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# MAGIC %md
# MAGIC # TRAIN MODELS

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark ML

# COMMAND ----------

# Load training data
training = spark.read.parquet(TRANSFORMED_TRAINING_DATA)
testing = spark.read.parquet(TRANSFORMED_2019_DATA)

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
input_cols = ['origin_weather_Present_Weather_Hail', 'origin_weather_Present_Weather_Storm', 'dest_weather_Present_Weather_Hail', 'dest_weather_Present_Weather_Storm', 'prior_delayed', '_QUARTER', '_MONTH', '_DAY_OF_WEEK', '_CRS_DEPT_HR', '_DISTANCE_GROUP', '_AIRLINE_DELAYS', '_origin_airport_type', '_dest_airport_type']
# input_cols = ['dest_weather_Present_Weather_Hail']
if 'features' in training.columns:
    training = training.drop('features')
training = VectorAssembler(inputCols=input_cols, outputCol="features").transform(training)

if 'features' in testing.columns:
    testing = testing.drop('features')
testing = VectorAssembler(inputCols=input_cols, outputCol="features").transform(testing)

# COMMAND ----------

def crossValidationSplit(df, CV_strategy=1):
    cv_splits = []
    if CV_strategy == 0:
        df_train = df.filter((df.YEAR <= 2017))
        df_test = df.filter((df.YEAR> 2017))
        cv_splits.append([df_train, df_test])
    elif CV_strategy == 1:
        df_2015 = df.filter((df.YEAR==2015))
        df_2016 = df.filter((df.YEAR==2016))
        cv_splits.append([df_2015, df_2016])
        df_2017 = df.filter((df.YEAR==2017))
        df_2018 = df.filter((df.YEAR==2018))
        cv_splits.append([df_2017, df_2018])
    return cv_splits

# COMMAND ----------

def run_logistic(df, features="features", label="DEP_DEL15", balance=1, iterations=100, regularizaion_params=0.3, eNet_params=0.8):
    
    negatives = df.filter(df[label]==0).count()
    positives = df.filter(df[label]==1).count()
    balance_ratio = negatives / (positives+negatives)
    df = df.withColumn("classWeights", when(df[label] == 1, balance_ratio).otherwise(1-balance_ratio))
    
    lr = LogisticRegression(featuresCol=features,
                           labelCol=label,
                           weightCol="classWeights",
                           maxIter=iterations,
                           regParam=regularization_params,
                           elasticNetParam=eNet_params)
    
    # Fit the model
    model = lr.fit(df)
    return model
    

# COMMAND ----------

def run_model(df, model='logistic', CV_strategy=1):
    
    cv_group = crossValidationSplit(df, CV_strategy)
    
    if CV_strategy == 0:   # no cross-validation
        train = cv_group[0][0]
        test = cv_group[0][1]
        lrModel = run_logistic(train)
        result = lrModel.transform(test).select('DEP_DEL15', 'prediction').withColumnRenamed('DEP_DEL15', 'label').cache()
        
    elif CV_strategy ==1:    # cross-validation with 2-folds, 2 years each, 50/50 split (1 year train, 1 year test)
        train_1 = cv_group[0][0]
        test_1 = cv_group[0][1]
        lrModel_1 = run_logistic(train_1)
        result_1 = lrModel_1.transform(test_1).select('DEP_DEL15', 'prediction').withColumnRenamed('DEP_DEL15', 'label').cache()

        train_2 = cv_group[1][0]
        test_2 = cv_group[1][1]
        lrModel_2 = run_logistic(train_2)
        result_2 = lrModel_2.transform(test_2).select('DEP_DEL15', 'prediction').withColumnRenamed('DEP_DEL15', 'label').cache()

        result = result_1.union(result_2)
        
    return result    

result_base = run_model(training, CV_strategy=0)
result_CV_1 = run_model(training, CV_strategy=1)


# COMMAND ----------

print(result_base.filter(result_base.prediction == 1).count(), result_CV_1.filter(result_CV_1.prediction == 1).count())

# COMMAND ----------

def model_performance(result, beta=1.0):
    performance = {}
    TP = result.filter((result.label == 1) & (result.prediction == 1)).count()
    TN = result.filter((result.label == 0) & (result.prediction == 0)).count()
    FP = result.filter((result.label == 0) & (result.prediction == 1)).count()
    FN = result.filter((result.label == 1) & (result.prediction == 0)).count()
    
    if (TP+TN+FP+FN) != 0:
        accuracy = (TP + TN) / (TP+TN+FP+FN)
    else:
        accuracy = 0
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    if ((beta**2) * precision + recall) != 0:
        f_beta = ((1+beta**2) * (precision * recall)) / ((beta**2) * precision + recall)
    else:
        f_beta = 0
    if (precision + recall) != 0:
        f_score = (precision * recall) / (precision + recall)
    else:
        f_score = 0
    
    performance['TP'] = TP
    performance['TN'] = TN
    performance['FP'] = FP
    performance['FN'] = FN
    performance['accuracy'] = accuracy
    performance['precision'] = precision
    performance['recall'] = recall
    performance['f_beta'] = f_beta
    performance['f_score'] = f_score
    return performance


perf_base = model_performance(result_base, beta=0.5)
print('BASE MODEL')
print(f"True Positives: {perf_base['TP']}")
print(f"True Negatives: {perf_base['TN']}")
print(f"False Positives: {perf_base['FP']}")
print(f"False Negatives: {perf_base['FN']}")
print(f"Accuracy: {perf_base['accuracy']}")
print(f"Precision: {perf_base['precision']}")
print(f"Recall: {perf_base['recall']}")
print(f"F_beta (beta=0.5): {perf_base['f_beta']}")
print(f"F_score (beta=1.0): {perf_base['f_score']}")

perf_CV_1 = model_performance(result_CV_1, beta=0.5)
print('------------------')
print('CV-1 MODEL')
print(f"True Positives: {perf_CV_1['TP']}")
print(f"True Negatives: {perf_CV_1['TN']}")
print(f"False Positives: {perf_CV_1['FP']}")
print(f"False Negatives: {perf_CV_1['FN']}")
print(f"Accuracy: {perf_CV_1['accuracy']}")
print(f"Precision: {perf_CV_1['precision']}")
print(f"Recall: {perf_CV_1['recall']}")
print(f"F_beta (beta=0.5): {perf_CV_1['f_beta']}")
print(f"F_score (beta=1.0): {perf_CV_1['f_score']}")


# COMMAND ----------

# Print the coefficients and intercept for logistic regression, and save for later
# SML_Coefficients = ast.literal_eval(str(lrModel.coefficients))
# SML_Intercept = lrModel.intercept
# print("Coefficients: " + str(lrModel.coefficients))
# print("Intercept: " + str(lrModel.intercept))

# COMMAND ----------


