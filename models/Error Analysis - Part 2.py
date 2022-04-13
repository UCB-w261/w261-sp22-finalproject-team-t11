# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Error Analysis
# MAGIC 
# MAGIC This notebook contains the error analysis performed on the main models run as part of our product development process.

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
from pyspark.sql.functions import col, isnan, when, count, lit, explode, array
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, LogisticRegression
from pyspark.ml.tuning import CrossValidatorModel

# magic commands
%matplotlib inline
%reload_ext autoreload
%autoreload 2

from configuration_v01 import Configuration
from data_preparer_v05 import DataPreparer
from training_summarizer_v01 import TrainingSummarizer
configuration = Configuration()
data_preparer = DataPreparer(configuration)
summarizer = TrainingSummarizer(configuration)

TRAINED_LR_MODEL_BASELINE = f'{configuration.TRAINED_MODEL_BASE_PATH}/lr_baseline'
TRAINED_LR_MODEL_GRIDSEARCH = f'{configuration.TRAINED_MODEL_BASE_PATH}/lr_gridsearch'
TRAINED_RF_MODEL_BASELINE = f'{configuration.TRAINED_MODEL_BASE_PATH}/rf_baseline'

# TRAINED_RF_MODEL_BASE = f'{configuration.TRAINED_MODEL_BASE_PATH}/rf_baseline_test'
# TRAINED_RF_MODEL_BASE = f'{configuration.TRAINED_MODEL_BASE_PATH}/rf_toy'

def load_model(location, modelClass):
    return modelClass.load(location)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Load Data and Models

# COMMAND ----------

# Load and Transform data
train = data_preparer.prep_data(spark.read.parquet(configuration.TRANSFORMED_TRAINING_DATA))
test_2019 = data_preparer.prep_data(spark.read.parquet(configuration.TRANSFORMED_2019_DATA))
test_2020 = data_preparer.prep_data(spark.read.parquet(configuration.TRANSFORMED_2020_DATA))
test_2021 = data_preparer.prep_data(spark.read.parquet(configuration.TRANSFORMED_2021_DATA))


# COMMAND ----------

# Load best models
model_lr = load_model(TRAINED_LR_MODEL_GRIDSEARCH, CrossValidatorModel).bestModel
model_rf = load_model(TRAINED_RF_MODEL_BASELINE, CrossValidatorModel).bestModel
model_lr_baseline = load_model(TRAINED_LR_MODEL_BASELINE, CrossValidatorModel).bestModel

# COMMAND ----------

configuration.feature_cols

# COMMAND ----------

features_cols = ['FL_DATE', 
                'OP_UNIQUE_CARRIER', 
                'OP_CARRIER_FL_NUM', 
                'DEP_DELAY_NEW', 
                'DEP_DELAY_GROUP', 
                'CRS_ELAPSED_TIME', 
                'DISTANCE',
                'ORIGIN_AIRPORT_ID', 
                'DEST_AIRPORT_ID',
                '_CRS_DEPT_HR',
                'origin_weather_Avg_Elevation',
                'origin_weather_Avg_HourlyDryBulbTemperature',
                'origin_weather_Avg_HourlyWindSpeed',
                'origin_weather_NonZero_Rain',
                'origin_weather_HourlyPressureTendency_Increasing',
                'origin_weather_Present_Weather_Hail', 
                'origin_weather_Present_Weather_Storm', 
                'dest_weather_Avg_Elevation',
                'dest_weather_Avg_HourlyDryBulbTemperature',
                'dest_weather_Avg_HourlyWindSpeed',
                'dest_weather_NonZero_Rain',
                'dest_weather_HourlyPressureTendency_Increasing',
                'dest_weather_Present_Weather_Hail',
                'dest_weather_Present_Weather_Storm',
                 'prior_delayed', 
                '_origin_airport_type', 
                '_dest_airport_type',
                'Holiday_5Day',
                'FalsePositive', 
                'label', 
                'prediction']

# COMMAND ----------

epsilon = 1e-5

def get_precision(tp, fp):
#     return tp / (tp+fp+epsilon)
    return tp / (tp+fp)

def get_recall(tp, fn):
#     return tp / (tp+fn+epsilon)
    return tp / (tp+fn)

def get_f1_score(tp, fp, fn):
#     return (2*tp) / (2*tp + fp + fn + epsilon)
    return (2*tp) / ((2*tp) + fp + fn)

def get_f_beta(precision, recall, beta=0.5):
#     return (1+ beta**2)*(precision*recall)/(beta**2 * precision + recall + epsilon)
    return (1+ (beta**2))*(precision*recall)/(((beta**2) * precision) + recall)

def get_accuracy(tp, tn, fp, fn):
    return (tp+tn) / (tp+tn+fp+fn)

def get_precision_w(tp_0, fp_0, tp_1, fp_1, w_0, w_1):
    p0 = (tp_0 / (tp_0+fp_0)) * w_0
    p1 = (tp_1 / (tp_1+fp_1)) * w_1
    return p0+p1

def get_recall_w(tp_0, fn_0, tp_1, fn_1, w_0, w_1):
    r0 = (tp_0 / (tp_0 + fn_0)) * w_0
    r1 = (tp_1 / (tp_1 + fn_1)) * w_1
    return r0+r1

def get_f1_score_w(precision, recall):
    return 2*(precision*recall) / (precision + recall)

def get_f_beta_w(precision, recall, beta=0.5):
    return (1+ (beta**2))*(precision*recall)/(((beta**2) * precision) + recall)

def get_accuracy_w(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1):
    return (tp_0+tp_1) / (tp_0+fp_0+tn_0+fn_0)

def count_w_data(df):
    
    total = df.count()

    tp_0 = df.select('label','prediction').where((df['label']==0) & (df['prediction']==0)).count()
    tn_0 = df.select('label','prediction').where((df['label']==1) & (df['prediction']==1)).count()
    fp_0 = df.select('label','prediction').where((df['label']==1) & (df['prediction']==0)).count()
    fn_0 = df.select('label','prediction').where((df['label']==0) & (df['prediction']==1)).count()


    tp_1 = df.select('label','prediction').where((df['label']==1) & (df['prediction']==1)).count()
    tn_1 = df.select('label','prediction').where((df['label']==0) & (df['prediction']==0)).count()
    fp_1 = df.select('label','prediction').where((df['label']==0) & (df['prediction']==1)).count()
    fn_1 = df.select('label','prediction').where((df['label']==1) & (df['prediction']==0)).count()

    support_0 = tp_0+fn_0  # Support for class 0 equals to all labels==0
    support_1 = tp_1+fn_1  # Support for class 1 equals to all labels==1
    w_0 = support_0 / total
    w_1 = support_1 / total
    
    return(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total)

def print_w_scores(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total):
    precision_w = get_precision_w(tp_0, fp_0, tp_1, fp_1, w_0, w_1)
    recall_w = get_recall_w(tp_0, fn_0, tp_1, fp_1, w_0, w_1)
    f1_score_w = get_f1_score_w(precision_w, recall_w)
    f_beta_w = get_f_beta_w(precision_w, recall_w)
    accuracy_w = get_accuracy_w(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1)

    print(f'Total # of flights:  {total}')
    print(f'True Positives (1):  {tp_1}')
    print(f'True Negatives (1):  {tn_1}')
    print(f'False Positives (1): {fp_1}')
    print(f'False Negatives (1): {fn_1}')
    print(f'Precision:           {precision_w}')
    print(f'Recall:              {recall_w}')
    print(f'F1-score:            {f1_score_w}')
    print(f'F_beta:              {f_beta_w}')
    print(f'Accuracy:            {accuracy_w}')
    
def print_scores(tp, fp, tn, fn, total):
    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn)
    f1_score = get_f1_score(tp, fp, fn)
    f_beta = get_f_beta(precision, recall)
    accuracy = get_accuracy(tp, tn, fp, fn)

    print(f'Total # of flights: {total}')
    print(f'True Positives:     {tp}')
    print(f'True Negatives:     {tn}')
    print(f'False Positives:    {fp}')
    print(f'False Negatives:    {fn}')
    print(f'Precision:          {precision}')
    print(f'Recall:             {recall}')
    print(f'F1-score:           {f1_score}')
    print(f'F_beta:             {f_beta}')
    print(f'Accuracy:           {accuracy}')

# COMMAND ----------

def get_matrix_1f(df, feature1):

    total = df.groupBy(feature1).count().withColumnRenamed('count', 'total_flights')

    tp_0 = df.select(feature1, 'label','prediction').where((df['label']==0) & (df['prediction']==0)) \
                    .groupBy(feature1).count().withColumnRenamed('count', 'tp_0')
    tn_0 = df.select(feature1, 'label','prediction').where((df['label']==1) & (df['prediction']==1)) \
                    .groupBy(feature1).count().withColumnRenamed('count', 'tn_0')
    fp_0 = df.select(feature1, 'label','prediction').where((df['label']==1) & (df['prediction']==0)) \
                    .groupBy(feature1).count().withColumnRenamed('count', 'fp_0')
    fn_0 = df.select(feature1, 'label','prediction').where((df['label']==0) & (df['prediction']==1)) \
                    .groupBy(feature1).count().withColumnRenamed('count', 'fn_0')

    tp_1 = df.select(feature1, 'label','prediction').where((df['label']==1) & (df['prediction']==1)) \
                    .groupBy(feature1).count().withColumnRenamed('count', 'tp_1')
    tn_1 = df.select(feature1, 'label','prediction').where((df['label']==0) & (df['prediction']==0)) \
                    .groupBy(feature1).count().withColumnRenamed('count', 'tn_1')
    fp_1 = df.select(feature1, 'label','prediction').where((df['label']==0) & (df['prediction']==1)) \
                    .groupBy(feature1).count().withColumnRenamed('count', 'fp_1')
    fn_1 = df.select(feature1, 'label','prediction').where((df['label']==1) & (df['prediction']==0)) \
                    .groupBy(feature1).count().withColumnRenamed('count', 'fn_1')

    support_0 = df.where(df.label==0).count()
    support_1 = df.where(df.label==1).count()
    tot_support = support_0 + support_1

    w_0 = support_0 / tot_support
    w_1 = support_1 / tot_support

    summary = total.join(fp_0, on=([feature1])) \
                   .join(fn_0, on=([feature1])) \
                   .join(tp_0, on=([feature1])) \
                   .join(tn_0, on=([feature1])) \
                   .join(fp_1, on=([feature1])) \
                   .join(fn_1, on=([feature1])) \
                   .join(tp_1, on=([feature1])) \
                   .join(tn_1, on=([feature1]))
    
    summary = summary.withColumn('precision_w', get_precision_w(col('tp_0'), col('fp_0'), col('tp_1'), col('fp_1'), w_0, w_1)) \
                        .withColumn('recall_w', get_recall_w(col('tp_0'), col('fn_0'), col('tp_1'), col('fn_1'), w_0, w_1)) \
                        .withColumn('f1_score_w', get_f1_score_w(col('precision_w'), col('recall_w'))) \
                        .withColumn('f_beta_w', get_f_beta_w(col('precision_w'), col('recall_w'))) \
                        .withColumn('accuracy_w', get_accuracy_w(col('tp_0'), col('tn_0'), col('fp_0'), col('fn_0'), col('tp_1'), col('tn_1'), col('fp_1'), col('fn_1'), w_0, w_1)) \
                        .withColumn('precision', get_precision(col('tp_1'), col('fp_1'))) \
                        .withColumn('recall', get_recall(col('tp_1'), col('fn_1'))) \
                        .withColumn('f1_score', get_f1_score(col('tp_1'), col('fp_1'), col('fn_1'))) \
                        .withColumn('f_beta', get_f_beta(col('precision'), col('recall'))) \
                        .withColumn('accuracy', get_accuracy(col('tp_1'), col('tn_1'), col('fp_1'), col('fn_1')))
    
    return summary.sort('f_beta_w')

def get_matrix_2f(df, feature1, feature2):

    total = df.groupBy(feature1, feature2).count().withColumnRenamed('count', 'total_flights')

    tp_0 = df.select(feature1, feature2, 'label','prediction').where((df['label']==0) & (df['prediction']==0)) \
                    .groupBy(feature1, feature2).count().withColumnRenamed('count', 'tp_0')
    tn_0 = df.select(feature1, feature2, 'label','prediction').where((df['label']==1) & (df['prediction']==1)) \
                    .groupBy(feature1, feature2).count().withColumnRenamed('count', 'tn_0')
    fp_0 = df.select(feature1, feature2, 'label','prediction').where((df['label']==1) & (df['prediction']==0)) \
                    .groupBy(feature1, feature2).count().withColumnRenamed('count', 'fp_0')
    fn_0 = df.select(feature1, feature2, 'label','prediction').where((df['label']==0) & (df['prediction']==1)) \
                    .groupBy(feature1, feature2).count().withColumnRenamed('count', 'fn_0')

    tp_1 = df.select(feature1, feature2, 'label','prediction').where((df['label']==1) & (df['prediction']==1)) \
                    .groupBy(feature1, feature2).count().withColumnRenamed('count', 'tp_1')
    tn_1 = df.select(feature1, feature2, 'label','prediction').where((df['label']==0) & (df['prediction']==0)) \
                    .groupBy(feature1, feature2).count().withColumnRenamed('count', 'tn_1')
    fp_1 = df.select(feature1, feature2, 'label','prediction').where((df['label']==0) & (df['prediction']==1)) \
                    .groupBy(feature1, feature2).count().withColumnRenamed('count', 'fp_1')
    fn_1 = df.select(feature1, feature2, 'label','prediction').where((df['label']==1) & (df['prediction']==0)) \
                    .groupBy(feature1, feature2).count().withColumnRenamed('count', 'fn_1')

    support_0 = df.where(df.label==0).count()
    support_1 = df.where(df.label==1).count()
    tot_support = support_0 + support_1

    w_0 = support_0 / tot_support
    w_1 = support_1 / tot_support

    summary = total.join(fp_0, on=([feature1, feature2])) \
                   .join(fn_0, on=([feature1, feature2])) \
                   .join(tp_0, on=([feature1, feature2])) \
                   .join(tn_0, on=([feature1, feature2])) \
                   .join(fp_1, on=([feature1, feature2])) \
                   .join(fn_1, on=([feature1, feature2])) \
                   .join(tp_1, on=([feature1, feature2])) \
                   .join(tn_1, on=([feature1, feature2]))
    
    summary = summary.withColumn('precision_w', get_precision_w(col('tp_0'), col('fp_0'), col('tp_1'), col('fp_1'), w_0, w_1)) \
                        .withColumn('recall_w', get_recall_w(col('tp_0'), col('fn_0'), col('tp_1'), col('fn_1'), w_0, w_1)) \
                        .withColumn('f1_score_w', get_f1_score_w(col('precision_w'), col('recall_w'))) \
                        .withColumn('f_beta_w', get_f_beta_w(col('precision_w'), col('recall_w'))) \
                        .withColumn('accuracy_w', get_accuracy_w(col('tp_0'), col('tn_0'), col('fp_0'), col('fn_0'), col('tp_1'), col('tn_1'), col('fp_1'), col('fn_1'), w_0, w_1)) \
                        .withColumn('precision', get_precision(col('tp_1'), col('fp_1'))) \
                        .withColumn('recall', get_recall(col('tp_1'), col('fn_1'))) \
                        .withColumn('f1_score', get_f1_score(col('tp_1'), col('fp_1'), col('fn_1'))) \
                        .withColumn('f_beta', get_f_beta(col('precision'), col('recall'))) \
                        .withColumn('accuracy', get_accuracy(col('tp_1'), col('tn_1'), col('fp_1'), col('fn_1')))

    return summary.sort('f_beta_w')


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Load Results (Logistic Regression)

# COMMAND ----------

# Get results for best LR model

train_pred_lr = model_lr.transform(train)
test_2019_pred_lr = model_lr.transform(test_2019)
test_2020_pred_lr = model_lr.transform(test_2020)
test_2021_pred_lr = model_lr.transform(test_2021)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Logistic Regression

# COMMAND ----------

train_pred = train_pred_lr.withColumn('FalsePositive', ((train_pred_lr['label']!=train_pred_lr['prediction']) & (train_pred_lr['prediction']==1)).cast('integer'))
test_2019_pred = test_2019_pred_lr.withColumn('FalsePositive', ((test_2019_pred_lr['label']!=test_2019_pred_lr['prediction']) & (test_2019_pred_lr['prediction']==1)).cast('integer'))
test_2020_pred = test_2020_pred_lr.withColumn('FalsePositive', ((test_2020_pred_lr['label']!=test_2020_pred_lr['prediction']) & (test_2020_pred_lr['prediction']==1)).cast('integer'))
test_2021_pred = test_2021_pred_lr.withColumn('FalsePositive', ((test_2021_pred_lr['label']!=test_2021_pred_lr['prediction']) & (test_2021_pred_lr['prediction']==1)).cast('integer'))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Overall Metrics

# COMMAND ----------

tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total = count_w_data(train_pred)
print('Weighted Metrics')
print_w_scores(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total)
print('\nBinary Metrics')
print_scores(tp_1, fp_1, tn_1, fn_1, total)

# COMMAND ----------

tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total = count_w_data(test_2019_pred)
print('Weighted Metrics')
print_w_scores(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total)
print('\nBinary Metrics')
print_scores(tp_1, fp_1, tn_1, fn_1, total)

# COMMAND ----------

tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total = count_w_data(test_2020_pred)
print('Weighted Metrics')
print_w_scores(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total)
print('\nBinary Metrics')
print_scores(tp_1, fp_1, tn_1, fn_1, total)

# COMMAND ----------

tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total = count_w_data(test_2021_pred)
print('Weighted Metrics')
print_w_scores(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total)
print('\nBinary Metrics')
print_scores(tp_1, fp_1, tn_1, fn_1, total)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Airport pairs

# COMMAND ----------

airport_pairs_19 = get_matrix_2f(test_2019_pred, 'ORIGIN', 'DEST')
display(airport_pairs_19)

# COMMAND ----------

airport_pairs_20 = get_matrix_2f(test_2020_pred, 'ORIGIN', 'DEST')
display(airport_pairs_20)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## ORIGIN airport

# COMMAND ----------

origin_airports_19 = get_matrix_1f(test_2019_pred, 'ORIGIN')
display(origin_airports_19)

# COMMAND ----------

origin_airports_20 = get_matrix_1f(test_2020_pred, 'ORIGIN')
display(origin_airports_20)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## DESTINATION airport

# COMMAND ----------

destination_airports_19 = get_matrix_1f(test_2019_pred, 'DEST')
display(destination_airports_19)

# COMMAND ----------

destination_airports_20 = get_matrix_1f(test_2020_pred, 'DEST')
display(destination_airports_20)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Airline

# COMMAND ----------

airlines_19 = get_matrix_1f(test_2019_pred, 'OP_UNIQUE_CARRIER')
display(airlines_19)

# COMMAND ----------

airlines_20 = get_matrix_1f(test_2020_pred, 'OP_UNIQUE_CARRIER')
display(airlines_20)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Delay groups

# COMMAND ----------

delay_groups_19 = get_matrix_1f(test_2019_pred, 'DEP_DELAY_GROUP')
display(delay_groups_19)

# COMMAND ----------

delay_groups_20 = get_matrix_1f(test_2020_pred, 'DEP_DELAY_GROUP')
display(delay_groups_20)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Departure time

# COMMAND ----------

departure_times_19 = get_matrix_1f(test_2019_pred, '_CRS_DEPT_HR')
display(departure_times_19)

# COMMAND ----------

departure_times_20 = get_matrix_1f(test_2020_pred, '_CRS_DEPT_HR')
display(departure_times_20)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## ORIGIN airport type

# COMMAND ----------

origin_airport_type_19 = get_matrix_1f(test_2019_pred, '_origin_airport_type')
display(origin_airport_type_19)

# COMMAND ----------

origin_airport_type_20 = get_matrix_1f(test_2020_pred, '_origin_airport_type')
display(origin_airport_type_20)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## DESTINATION airport type

# COMMAND ----------

dest_airport_type_19 = get_matrix_1f(test_2019_pred, '_dest_airport_type')
display(dest_airport_type_19)

# COMMAND ----------

dest_airport_type_20 = get_matrix_1f(test_2020_pred, '_dest_airport_type')
display(dest_airport_type_20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Incoming aircraft delay

# COMMAND ----------

prior_delayed_19 = get_matrix_1f(test_2019_pred, 'prior_delayed')
display(prior_delayed_19)

# COMMAND ----------

prior_delayed_20 = get_matrix_1f(test_2020_pred, 'prior_delayed')
display(prior_delayed_20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Storm at ORIGIN airport

# COMMAND ----------

storm_origin_19 = get_matrix_1f(test_2019_pred, 'origin_weather_Present_Weather_Storm')
display(storm_origin_19)

# COMMAND ----------

storm_origin_20 = get_matrix_1f(test_2020_pred, 'origin_weather_Present_Weather_Storm')
display(storm_origin_20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Storm at DESTINATION airport

# COMMAND ----------

storm_dest_19 = get_matrix_1f(test_2019_pred, 'dest_weather_Present_Weather_Storm')
display(storm_dest_19)

# COMMAND ----------

storm_dest_20 = get_matrix_1f(test_2020_pred, 'dest_weather_Present_Weather_Storm')
display(storm_dest_20)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Random Forest - Baseline

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Load Results (Random Forest)

# COMMAND ----------

# Get results for best RF model

train_pred_rf = model_rf.transform(train)
test_2019_pred_rf = model_rf.transform(test_2019)
test_2020_pred_rf = model_rf.transform(test_2020)
test_2021_pred_rf = model_rf.transform(test_2021)


# COMMAND ----------

rf_train_pred = train_pred_rf.withColumn('FalsePositive', ((train_pred_rf['label']!=train_pred_rf['prediction']) & (train_pred_rf['prediction']==1)).cast('integer'))
rf_test_2019_pred = test_2019_pred_rf.withColumn('FalsePositive', ((test_2019_pred_rf['label']!=test_2019_pred_rf['prediction']) & (test_2019_pred_rf['prediction']==1)).cast('integer'))
rf_test_2020_pred = test_2020_pred_rf.withColumn('FalsePositive', ((test_2020_pred_rf['label']!=test_2020_pred_rf['prediction']) & (test_2020_pred_rf['prediction']==1)).cast('integer'))
rf_test_2021_pred = test_2021_pred_rf.withColumn('FalsePositive', ((test_2021_pred_rf['label']!=test_2021_pred_rf['prediction']) & (test_2021_pred_rf['prediction']==1)).cast('integer'))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Overall Metrics

# COMMAND ----------

tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total = count_w_data(rf_train_pred)
print('Weighted Metrics')
print_w_scores(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total)
print('\nBinary Metrics')
print_scores(tp_1, fp_1, tn_1, fn_1, total)

# COMMAND ----------

tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total = count_w_data(rf_test_2019_pred)
print('Weighted Metrics')
print_w_scores(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total)
print('\nBinary Metrics')
print_scores(tp_1, fp_1, tn_1, fn_1, total)

# COMMAND ----------

tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total = count_w_data(rf_test_2020_pred)
print('Weighted Metrics')
print_w_scores(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total)
print('\nBinary Metrics')
print_scores(tp_1, fp_1, tn_1, fn_1, total)

# COMMAND ----------

tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total = count_w_data(rf_test_2021_pred)
print('Weighted Metrics')
print_w_scores(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total)
print('\nBinary Metrics')
print_scores(tp_1, fp_1, tn_1, fn_1, total)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Logistic Regression - Baseline

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Load Results (Logistic Regression - Baseline)

# COMMAND ----------

# Get results for best RF model

train_pred_lr_base = model_lr_baseline.transform(train)
test_2019_pred_lr_base = model_lr_baseline.transform(test_2019)
test_2020_pred_lr_base = model_lr_baseline.transform(test_2020)
test_2021_pred_lr_base = model_lr_baseline.transform(test_2021)


# COMMAND ----------

lr_base_train_pred = train_pred_lr_base.withColumn('FalsePositive', ((train_pred_lr_base['label']!=train_pred_lr_base['prediction']) & (train_pred_lr_base['prediction']==1)).cast('integer'))
lr_base_test_2019_pred = train_pred_lr_base.withColumn('FalsePositive', ((train_pred_lr_base['label']!=train_pred_lr_base['prediction']) & (train_pred_lr_base['prediction']==1)).cast('integer'))
lr_base_test_2020_pred = train_pred_lr_base.withColumn('FalsePositive', ((train_pred_lr_base['label']!=train_pred_lr_base['prediction']) & (train_pred_lr_base['prediction']==1)).cast('integer'))
lr_base_test_2021_pred = train_pred_lr_base.withColumn('FalsePositive', ((train_pred_lr_base['label']!=train_pred_lr_base['prediction']) & (train_pred_lr_base['prediction']==1)).cast('integer'))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Overall Metrics

# COMMAND ----------

tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total = count_w_data(lr_base_train_pred)
print('Weighted Metrics')
print_w_scores(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total)
print('\nBinary Metrics')
print_scores(tp_1, fp_1, tn_1, fn_1, total)

# COMMAND ----------

tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total = count_w_data(lr_base_test_2019_pred)
print('Weighted Metrics')
print_w_scores(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total)
print('\nBinary Metrics')
print_scores(tp_1, fp_1, tn_1, fn_1, total)

# COMMAND ----------

tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total = count_w_data(lr_base_test_2020_pred)
print('Weighted Metrics')
print_w_scores(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total)
print('\nBinary Metrics')
print_scores(tp_1, fp_1, tn_1, fn_1, total)

# COMMAND ----------

tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total = count_w_data(lr_base_test_2021_pred)
print('Weighted Metrics')
print_w_scores(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total)
print('\nBinary Metrics')
print_scores(tp_1, fp_1, tn_1, fn_1, total)

# COMMAND ----------



# COMMAND ----------


