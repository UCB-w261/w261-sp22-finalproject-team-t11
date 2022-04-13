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

# FINAL_JOINED_DATA_TRAINING = f"{blob_url}/staged/final_joined_training"
# FINAL_JOINED_DATA_2019 = f"{blob_url}/staged/final_joined_testing"
# FINAL_JOINED_DATA_2020 = f"{blob_url}/staged/final_joined_2020"
# FINAL_JOINED_DATA_2021 = f"{blob_url}/staged/final_joined_2021"

# TRANSFORMED_TRAINING_DATA = f"{blob_url}/transformed/training"
# TRANSFORMED_2019_DATA = f"{blob_url}/transformed/2019"
# TRANSFORMED_2020_DATA = f"{blob_url}/transformed/2020"
# TRANSFORMED_2021_DATA = f"{blob_url}/transformed/2021"

# TRAINED_MODEL_BASE_PATH = f'{blob_url}/model/trained'

# SHAPES_BASE_FOLDER = "/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/shapes"

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

# orig_label_col = 'DEP_DEL15'
# label_col = 'label'
# features_col = 'features'

from configuration_v01 import Configuration
from data_preparer_v05 import DataPreparer
from training_summarizer_v01 import TrainingSummarizer
configuration = Configuration()
data_preparer = DataPreparer(configuration)
summarizer = TrainingSummarizer(configuration)

TRAINED_LR_MODEL_BASELINE = f'{configuration.TRAINED_MODEL_BASE_PATH}/lr_baseline'
TRAINED_LR_MODEL_GRIDSEARCH = f'{configuration.TRAINED_MODEL_BASE_PATH}/lr_gridsearch'
TRAINED_RF_MODEL_BASELINE = f'{configuration.TRAINED_MODEL_BASE_PATH}/rf_baseline'

MODEL_PREDICTIONS_BASE_PATH = f'{configuration.TRAINED_MODEL_BASE_PATH}/predictions'

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
model_lr_baseline = load_model(TRAINED_LR_MODEL_BASELINE, CrossValidatorModel).bestModel
model_rf = load_model(TRAINED_RF_MODEL_BASELINE, CrossValidatorModel).bestModel

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
                'dest_weather_Present_Weather_Storm','prior_delayed', 
                '_origin_airport_type', 
                '_dest_airport_type',
                'Holiday_5Day',
                'FalsePositive', 
                'label', 
                'prediction']

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

fp_train_lr = train_pred_lr.filter(train_pred_lr['label']!=train_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (train_pred_lr['prediction']==1).cast('integer')) \
                    .select(features_cols)
fp_test_2019_lr = test_2019_pred_lr.filter(test_2019_pred_lr['label']!=test_2019_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (test_2019_pred_lr['prediction']==1).cast('integer')) \
                    .select(features_cols)
fp_test_2020_lr = test_2020_pred_lr.filter(test_2020_pred_lr['label']!=test_2020_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (test_2020_pred_lr['prediction']==1).cast('integer')) \
                    .select(features_cols)
fp_test_2021_lr = test_2021_pred_lr.filter(test_2021_pred_lr['label']!=test_2021_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (test_2021_pred_lr['prediction']==1).cast('integer')) \
                    .select(features_cols)

fn_train_lr = train_pred_lr.filter(train_pred_lr['label']!=train_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (train_pred_lr['prediction']==0).cast('integer')) \
                    .select(features_cols)
fn_test_2019_lr = test_2019_pred_lr.filter(test_2019_pred_lr['label']!=test_2019_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (test_2019_pred_lr['prediction']==0).cast('integer')) \
                    .select(features_cols)
fn_test_2020_lr = test_2020_pred_lr.filter(test_2020_pred_lr['label']!=test_2020_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (test_2020_pred_lr['prediction']==0).cast('integer')) \
                    .select(features_cols)
fn_test_2021_lr = test_2021_pred_lr.filter(test_2021_pred_lr['label']!=test_2021_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (test_2021_pred_lr['prediction']==0).cast('integer')) \
                    .select(features_cols)

tp_train_lr = train_pred_lr.filter(train_pred_lr['label']==train_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (train_pred_lr['prediction']==1).cast('integer')) \
                    .select(features_cols)
tp_test_2019_lr = test_2019_pred_lr.filter(test_2019_pred_lr['label']==test_2019_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (test_2019_pred_lr['prediction']==1).cast('integer')) \
                    .select(features_cols)
tp_test_2020_lr = test_2020_pred_lr.filter(test_2020_pred_lr['label']==test_2020_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (test_2020_pred_lr['prediction']==1).cast('integer')) \
                    .select(features_cols)
tp_test_2021_lr = test_2021_pred_lr.filter(test_2021_pred_lr['label']==test_2021_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (test_2021_pred_lr['prediction']==1).cast('integer')) \
                    .select(features_cols)

tn_train_lr = train_pred_lr.filter(train_pred_lr['label']==train_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (train_pred_lr['prediction']==0).cast('integer')) \
                    .select(features_cols)
tn_test_2019_lr = test_2019_pred_lr.filter(test_2019_pred_lr['label']==test_2019_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (test_2019_pred_lr['prediction']==0).cast('integer')) \
                    .select(features_cols)
tn_test_2020_lr = test_2020_pred_lr.filter(test_2020_pred_lr['label']==test_2020_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (test_2020_pred_lr['prediction']==0).cast('integer')) \
                    .select(features_cols)
tn_test_2021_lr = test_2021_pred_lr.filter(test_2021_pred_lr['label']==test_2021_pred_lr['prediction']) \
                    .withColumn('FalsePositive', (test_2021_pred_lr['prediction']==0).cast('integer')) \
                    .select(features_cols)

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

fp_train_rf = train_pred_rf.filter(train_pred_rf['label']!=train_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (train_pred_rf['prediction']==1).cast('integer')) \
                    .select(features_cols)
fp_test_2019_rf = test_2019_pred_rf.filter(test_2019_pred_rf['label']!=test_2019_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (test_2019_pred_rf['prediction']==1).cast('integer')) \
                    .select(features_cols)
fp_test_2020_rf = test_2020_pred_rf.filter(test_2020_pred_rf['label']!=test_2020_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (test_2020_pred_rf['prediction']==1).cast('integer')) \
                    .select(features_cols)
fp_test_2021_rf = test_2021_pred_rf.filter(test_2021_pred_rf['label']!=test_2021_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (test_2021_pred_rf['prediction']==1).cast('integer')) \
                    .select(features_cols)

fn_train_rf = train_pred_rf.filter(train_pred_rf['label']!=train_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (train_pred_rf['prediction']==0).cast('integer')) \
                    .select(features_cols)
fn_test_2019_rf = test_2019_pred_rf.filter(test_2019_pred_rf['label']!=test_2019_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (test_2019_pred_rf['prediction']==0).cast('integer')) \
                    .select(features_cols)
fn_test_2020_rf = test_2020_pred_rf.filter(test_2020_pred_rf['label']!=test_2020_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (test_2020_pred_rf['prediction']==0).cast('integer')) \
                    .select(features_cols)
fn_test_2021_rf = test_2021_pred_rf.filter(test_2021_pred_rf['label']!=test_2021_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (test_2021_pred_rf['prediction']==0).cast('integer')) \
                    .select(features_cols)

tp_train_rf = train_pred_rf.filter(train_pred_rf['label']==train_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (train_pred_rf['prediction']==1).cast('integer')) \
                    .select(features_cols)
tp_test_2019_rf = test_2019_pred_rf.filter(test_2019_pred_rf['label']==test_2019_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (test_2019_pred_rf['prediction']==1).cast('integer')) \
                    .select(features_cols)
tp_test_2020_rf = test_2020_pred_rf.filter(test_2020_pred_rf['label']==test_2020_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (test_2020_pred_rf['prediction']==1).cast('integer')) \
                    .select(features_cols)
tp_test_2021_rf = test_2021_pred_rf.filter(test_2021_pred_rf['label']==test_2021_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (test_2021_pred_rf['prediction']==1).cast('integer')) \
                    .select(features_cols)

tn_train_rf = train_pred_rf.filter(train_pred_rf['label']==train_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (train_pred_rf['prediction']==0).cast('integer')) \
                    .select(features_cols)
tn_test_2019_rf = test_2019_pred_rf.filter(test_2019_pred_rf['label']==test_2019_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (test_2019_pred_rf['prediction']==0).cast('integer')) \
                    .select(features_cols)
tn_test_2020_rf = test_2020_pred_rf.filter(test_2020_pred_rf['label']==test_2020_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (test_2020_pred_rf['prediction']==0).cast('integer')) \
                    .select(features_cols)
tn_test_2021_rf = test_2021_pred_rf.filter(test_2021_pred_rf['label']==test_2021_pred_rf['prediction']) \
                    .withColumn('FalsePositive', (test_2021_pred_rf['prediction']==0).cast('integer')) \
                    .select(features_cols)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Correlation Matrix Analysis

# COMMAND ----------

corr_features = ['DEP_DELAY_NEW', 
                'DEP_DELAY_GROUP', 
                'CRS_ELAPSED_TIME', 
                'DISTANCE',
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
                'dest_weather_Present_Weather_Storm','prior_delayed', 
                '_origin_airport_type', 
                '_dest_airport_type',
                'Holiday_5Day',
                'FalsePositive', 
                'label', 
                'prediction']



# COMMAND ----------

def get_corr_matrix(df, features):
    
    df = df.select(features)
    vector_col = "corr_features"
    assembler = VectorAssembler(inputCols=features, outputCol=vector_col)
    df_vector = assembler.transform(df).select(vector_col)
    corr_matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
    corr_matrix = corr_matrix.toArray().tolist()
    corr_matrix_df = spark.createDataFrame(corr_matrix, features)

    return corr_matrix_df

# COMMAND ----------

def print_corr_matrix(matrix, title='Correlation Matrix'):
    _matrix = matrix.toPandas()
    size = int(_matrix.shape[0]/3*2+1)
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(size,size))
    fig.suptitle(title, ha='right', fontsize=24)
    mask = np.zeros_like(_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(_matrix, annot=True, fmt='.2f', mask=mask, cmap=cmap, ax=ax)
    ax.set_yticklabels(_matrix.columns)
    ax.tick_params(axis='y', labelrotation=0)
    # sns.heatmap(corr_matrix, cmap=cmap, xticklabels=corr_features, yticklabels=corr_features)

    fig.tight_layout()

def print_corr_bar(matrix, target_label = 'FalsePositive', title='Correlation Matrix'):
    _matrix = matrix.toPandas()
    size = int(_matrix.shape[0])
    fp_corr = pd.DataFrame(_matrix[target_label]).T

    fig, ax = plt.subplots(figsize=(size,6))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(fp_corr, annot=True, fmt='.2f', cmap=cmap, ax=ax, cbar=False, yticklabels=False)
    ax.set_xticklabels(_matrix.columns)
    ax.tick_params(axis='x', labelrotation=90)
    fig.tight_layout()
    

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Logistic Regression

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Training Data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(fp_train_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - False Positive Correlation Matrix on Train Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(tp_train_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - True Positive Correlation Matrix on Train Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(fn_train_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - False Negative Correlation Matrix on Train Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(tn_train_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - True Negative Correlation Matrix on Train Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2019 Test Data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(fp_test_2019_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - False Positive Correlation Matrix on 2019 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(tp_test_2019_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - True Positive Correlation Matrix on 2019 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(fn_test_2019_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - False Negative Correlation Matrix on 2019 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(tn_test_2019_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - True Negative Correlation Matrix on 2019 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2020 Test Data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(fp_test_2020_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - False Positive Correlation Matrix on 2020 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(tp_test_2020_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - True Positive Correlation Matrix on 2020 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(fn_test_2020_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - False Negative Correlation Matrix on 2020 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(tn_test_2020_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - True Negative Correlation Matrix on 2020 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2021 Test Data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(fp_test_2021_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - False Positive Correlation Matrix on 2021 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(tp_test_2021_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - True Positive Correlation Matrix on 2021 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(fn_test_2021_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - False Negative Correlation Matrix on 2021 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(tn_test_2021_lr, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Logistic Regression - True Negative Correlation Matrix on 2021 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Training Data

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### False Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(tp_train_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - False Positive Correlation Matrix on Train Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(tp_train_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - True Positive Correlation Matrix on Train Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(fn_train_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - False Negative Correlation Matrix on Train Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(tn_train_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - True Negative Correlation Matrix on Train Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2019 Test Data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(fp_test_2019_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - False Positive Correlation Matrix on 2019 Test DataSet')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(tp_test_2019_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - True Positive Correlation Matrix on 2019 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(fn_test_2019_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - False Negative Correlation Matrix on 2019 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(tn_test_2019_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - True Negative Correlation Matrix on 2019 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2020 Test Data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(fp_test_2020_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - False Positive Correlation Matrix on 2020 Test DataSet')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(tp_test_2020_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - True Positive Correlation Matrix on 2020 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(fn_test_2020_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - False Negative Correlation Matrix on 2020 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(tn_test_2020_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - True Negative Correlation Matrix on 2020 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2021 Test Data

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(fp_test_2021_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - False Positive Correlation Matrix on 2021 Test DataSet')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Positives

# COMMAND ----------

corr_matrix = get_corr_matrix(tp_test_2021_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - True Positive Correlation Matrix on 2021 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### False Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(fn_test_2021_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - False Negative Correlation Matrix on 2021 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### True Negatives

# COMMAND ----------

corr_matrix = get_corr_matrix(tn_test_2021_rf, corr_features)
print_corr_bar(corr_matrix, target_label='FalsePositive', title='Random Forest - True Negative Correlation Matrix on 2021 Test Data')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Summary Metrics

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Logistic Regression - Grid Search

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2019

# COMMAND ----------

summarizer.get_scores(test_2019_pred_lr)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2020

# COMMAND ----------

summarizer.get_scores(test_2020_pred_lr)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2021

# COMMAND ----------

summarizer.get_scores(test_2021_pred_lr)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Random Forest - Baseline

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2019

# COMMAND ----------

summarizer.get_scores(test_2019_pred_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2020

# COMMAND ----------

summarizer.get_scores(test_2020_pred_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2021

# COMMAND ----------

summarizer.get_scores(test_2021_pred_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Logistic Regression - Baseline

# COMMAND ----------

train_pred_lr_base = model_lr_baseline.transform(train)
test_2019_pred_lr_base = model_lr_baseline.transform(test_2019)
test_2020_pred_lr_base = model_lr_baseline.transform(test_2020)
test_2021_pred_lr_base = model_lr_baseline.transform(test_2021)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2019

# COMMAND ----------

summarizer.get_scores(test_2019_pred_lr_base)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2020

# COMMAND ----------

summarizer.get_scores(test_2020_pred_lr_base)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 2021

# COMMAND ----------

summarizer.get_scores(test_2021_pred_lr_base)

# COMMAND ----------

# Save predictions for later use.

def save_predictions(model_name, pred_train, pred_2019, pred_2020, pred_2021):
    pred_train.write.mode('overwrite').parquet(f'{MODEL_PREDICTIONS_BASE_PATH}/{model_name}/train')
    pred_2019.write.mode('overwrite').parquet(f'{MODEL_PREDICTIONS_BASE_PATH}/{model_name}/2019')
    pred_2020.write.mode('overwrite').parquet(f'{MODEL_PREDICTIONS_BASE_PATH}/{model_name}/2020')
    pred_2021.write.mode('overwrite').parquet(f'{MODEL_PREDICTIONS_BASE_PATH}/{model_name}/2021')

# LR Baseline
save_predictions('lr_baseline', train_pred_lr_base, test_2019_pred_lr_base, test_2020_pred_lr_base, test_2021_pred_lr_base)

# LR CV
save_predictions('lr_gridsearch', train_pred_lr, test_2019_pred_lr, test_2020_pred_lr, test_2021_pred_lr)

# RF Model
save_predictions('rf_baseline', train_pred_rf, test_2019_pred_rf, test_2020_pred_rf, test_2021_pred_rf)


# COMMAND ----------


