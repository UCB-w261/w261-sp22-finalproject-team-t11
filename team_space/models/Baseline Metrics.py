# Databricks notebook source
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
configuration = Configuration()

# COMMAND ----------


results_cols = ['Metric_type', 'Prediction', 'Year', 'Precision', 'Recall', 'F1', 'F_beta', 'Accuracy']
results = pd.DataFrame(columns = results_cols)

def calculate_metrics(tp, tn, fp, fn):
    epsilon = 1e-5
    beta = 0.5
    precision = tp / (tp+fp+epsilon)
    recall = tp / (tp+fn+epsilon)
    f1_score = (2*tp) / (2*tp + fp + fn + epsilon)
    f_beta = (1+ beta**2)*(precision*recall)/(beta**2 * precision + recall + epsilon)
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    return precision, recall, f1_score, f_beta, accuracy

def calculate_w_metrics(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total):
    epsilon = 1e-5
    beta = 0.5
    p0 = (tp_0 / (tp_0+fp_0+epsilon)) * w_0
    p1 = (tp_1 / (tp_1+fp_1+epsilon)) * w_1
    precision_w = p0+p1

    r0 = (tp_0 / (tp_0 + fn_0+epsilon)) * w_0
    r1 = (tp_1 / (tp_1 + fn_1+epsilon)) * w_1
    recall_w = r0+r1
    f1_score_w = 2*(precision_w*recall_w) / (precision_w + recall_w+epsilon)
    f_beta_w = (1+ (beta**2))*(precision_w*recall_w)/(((beta**2) * precision_w) + recall_w)
    accuracy_w = (tp_0+tp_1) / (tp_0+fp_0+tn_0+fn_0)

    return precision_w, recall_w, f1_score_w, f_beta_w, accuracy_w
 
    
def print_scores(tp_0, fp_0, tn_0, fn_0, tp_1, fp_1, tn_1, fn_1, title=None, prediction=None, year=None, results_df=None):

    results_cols = ['Metric_type', 'Prediction', 'Year', 'Precision', 'Recall', 'F1', 'F_beta', 'Accuracy']

    support_0 = tp_0+fn_0  # Support for class 0 equals to all labels==0
    support_1 = tn_0+fp_0  # Support for class 1 equals to all labels==1
    total = support_0 + support_1
    w_0 = support_0 / total
    w_1 = support_1 / total

    precision, recall, f1_score, f_beta, accuracy = calculate_metrics(tp_1, tn_1, fp_1, fn_1)
    precision_w, recall_w, f1_score_w, f_beta_w, accuracy_w = calculate_w_metrics(tp_0, tn_0, fp_0, fn_0, tp_1, tn_1, fp_1, fn_1, w_0, w_1, total)
    
    binary = ['binary', prediction, year, precision, recall, f1_score, f_beta, accuracy]
    weighted = ['weighted', prediction, year, precision_w, recall_w, f1_score_w, f_beta_w, accuracy_w]
    partial = pd.DataFrame([binary], columns=results_cols)
    partial_w = pd.DataFrame([weighted], columns=results_cols)
    results_df = pd.concat([results_df, partial], axis=0)
    results_df = pd.concat([results_df, partial_w], axis=0)
    
    print ("==============================================")
    print (f"{title} - WEIGHTED")
    print ("==============================================")
    print (f" Precision:       {precision_w:.4f}")
    print (f" Recall:          {recall_w:.4f}")
    print (f" F1 Score:        {f1_score_w:.4f}")
    print (f" F-beta (0.5):    {f_beta_w:.4f}")
    print (f" Accuracy:        {accuracy_w:.4f}")
    print ("==============================================")
    print (f"{title} - BINARY")
    print ("==============================================")
    print (f" Precision:       {precision:.4f}")
    print (f" Recall:          {recall:.4f}")
    print (f" F1 Score:        {f1_score:.4f}")
    print (f" F-beta (0.5):    {f_beta:.4f}")
    print (f" Accuracy:        {accuracy:.4f}")
    
    return results_df

# COMMAND ----------

df_train = spark.read.parquet(configuration.FINAL_JOINED_DATA_2015_2018)
df_test_2019 = spark.read.parquet(configuration.FINAL_JOINED_DATA_2019)
df_test_2020 = spark.read.parquet(configuration.FINAL_JOINED_DATA_2020)
df_test_2021 = spark.read.parquet(configuration.FINAL_JOINED_DATA_2021)
# df_full = df_train.union(df_test_2019)

# COMMAND ----------

total_train = df_train.count()
total_test_2019 = df_test_2019.count()
total_test_2020 = df_test_2020.count()
total_test_2021 = df_test_2021.count()

positives_train = df_train.filter(df_train['DEP_DEL15']==1).count()
negatives_train = df_train.filter(df_train['DEP_DEL15']==0).count()
invalid_train = df_train.filter(df_train['DEP_DEL15'].isNull()).count()
valid_train = total_train - invalid_train

positives_test_2019 = df_test_2019.filter(df_test_2019['DEP_DEL15']==1).count()
negatives_test_2019 = df_test_2019.filter(df_test_2019['DEP_DEL15']==0).count()
invalid_test_2019 = df_test_2019.filter(df_test_2019['DEP_DEL15'].isNull()).count()
valid_test_2019 = total_test_2019 - invalid_test_2019

positives_test_2020 = df_test_2020.filter(df_test_2020['DEP_DEL15']==1).count()
negatives_test_2020 = df_test_2020.filter(df_test_2020['DEP_DEL15']==0).count()
invalid_test_2020 = df_test_2020.filter(df_test_2020['DEP_DEL15'].isNull()).count()
valid_test_2020 = total_test_2020 - invalid_test_2020

positives_test_2021 = df_test_2021.filter(df_test_2021['DEP_DEL15']==1).count()
negatives_test_2021 = df_test_2021.filter(df_test_2021['DEP_DEL15']==0).count()
invalid_test_2021 = df_test_2021.filter(df_test_2021['DEP_DEL15'].isNull()).count()
valid_test_2021 = total_test_2021 - invalid_test_2021


# COMMAND ----------

results

# COMMAND ----------

# All positives scenario train (2015-2018)

tp_1 = positives_train
fp_1 = negatives_train
tn_1 = 0
fn_1 = 0

tp_0 = 0
fp_0 = 0
tn_0 = positives_train
fn_0 = negatives_train

results = pd.concat([results, print_scores(tp_0, fp_0, tn_0, fn_0, tp_1, fp_1, tn_1, fn_1, title='ALL POSITIVE PREDICTIONS (2015-18)', prediction='Minority', year='2015-18', results_df=results)])

# COMMAND ----------

# All negatives scenario train (2015-2018)

tp_0 = negatives_train
fp_0 = positives_train
tn_0 = 0
fn_0 = 0

tp_1 = 0
fp_1 = 0
tn_1 = negatives_train
fn_1 = positives_train

results = pd.concat([results, print_scores(tp_0, fp_0, tn_0, fn_0, tp_1, fp_1, tn_1, fn_1, title='ALL NEGATIVE PREDICTIONS (2015-18)', prediction='Majority', year='2015-18')])

# COMMAND ----------

# All positives scenario train (2019)

tp_1 = positives_test_2019
fp_1 = negatives_test_2019
tn_1 = 0
fn_1 = 0

tp_0 = 0
fp_0 = 0
tn_0 = positives_test_2019
fn_0 = negatives_test_2019

results = pd.concat([results, print_scores(tp_0, fp_0, tn_0, fn_0, tp_1, fp_1, tn_1, fn_1, title='ALL POSITIVE PREDICTIONS (2019)', prediction='Minority', year='2019')])


# COMMAND ----------

# All negatives scenario train (2019)

tp_0 = negatives_test_2019
fp_0 = positives_test_2019
tn_0 = 0
fn_0 = 0

tp_1 = 0
fp_1 = 0
tn_1 = negatives_test_2019
fn_1 = positives_test_2019

results = pd.concat([results, print_scores(tp_0, fp_0, tn_0, fn_0, tp_1, fp_1, tn_1, fn_1, title='ALL NEGATIVE PREDICTIONS (2019)', prediction='Majority', year='2019')])

# COMMAND ----------

# All positives scenario train (2020)

tp_1 = positives_test_2020
fp_1 = negatives_test_2020
tn_1 = 0
fn_1 = 0

tp_0 = 0
fp_0 = 0
tn_0 = positives_test_2020
fn_0 = negatives_test_2020

results = pd.concat([results, print_scores(tp_0, fp_0, tn_0, fn_0, tp_1, fp_1, tn_1, fn_1, title='ALL POSITIVE PREDICTIONS (2020)', prediction='Minority', year='2020')])


# COMMAND ----------

# All negatives scenario train (2020)

tp_0 = negatives_test_2020
fp_0 = positives_test_2020
tn_0 = 0
fn_0 = 0

tp_1 = 0
fp_1 = 0
tn_1 = negatives_test_2020
fn_1 = positives_test_2020

results = pd.concat([results, print_scores(tp_0, fp_0, tn_0, fn_0, tp_1, fp_1, tn_1, fn_1, title='ALL NEGATIVE PREDICTIONS (2020)', prediction='Majority', year='2020')])

# COMMAND ----------

# All positives scenario train (2021)

tp_1 = positives_test_2021
fp_1 = negatives_test_2021
tn_1 = 0
fn_1 = 0

tp_0 = 0
fp_0 = 0
tn_0 = positives_test_2021
fn_0 = negatives_test_2021

results = pd.concat([results, print_scores(tp_0, fp_0, tn_0, fn_0, tp_1, fp_1, tn_1, fn_1, title='ALL POSITIVE PREDICTIONS (2021)', prediction='Minority', year='2021')])


# COMMAND ----------

# All negatives scenario train (2021)

tp_0 = negatives_test_2021
fp_0 = positives_test_2021
tn_0 = 0
fn_0 = 0

tp_1 = 0
fp_1 = 0
tn_1 = negatives_test_2021
fn_1 = positives_test_2021

results = pd.concat([results, print_scores(tp_0, fp_0, tn_0, fn_0, tp_1, fp_1, tn_1, fn_1, title='ALL NEGATIVE PREDICTIONS (2021)', prediction='Majority', year='2021')])

# COMMAND ----------

results

# COMMAND ----------

results_formatted = results
results_formatted['Precision'] = results_formatted['Precision'].map('{:,.2f}'.format)
results_formatted['F1'] = results_formatted['F1'].map('{:,.2f}'.format)
results_formatted['F_beta'] = results_formatted['F_beta'].map('{:,.2f}'.format)
results_formatted['Recall'] = results_formatted['Recall'].map('{:,.2f}'.format)
results_formatted['Accuracy'] = results_formatted['Accuracy'].map('{:,.2f}'.format)


# COMMAND ----------

results_formatted

# COMMAND ----------

results_formatted[(results_formatted['Metric_type']=='weighted') & (results_formatted['Prediction']=='Majority')].T

# COMMAND ----------

results_formatted[(results_formatted['Metric_type']=='binary') & (results_formatted['Prediction']=='Majority')].T

# COMMAND ----------

results_formatted[(results_formatted['Metric_type']=='weighted') & (results_formatted['Prediction']=='Minority')].T

# COMMAND ----------

results_formatted[(results_formatted['Metric_type']=='binary') & (results_formatted['Prediction']=='Minority')].T

# COMMAND ----------


