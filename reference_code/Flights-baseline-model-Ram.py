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

from pyspark.sql.functions import col, isnan, when, count

# COMMAND ----------

label_col = "DEP_DEL15"
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

class_weights_col = 'classWeights'
num_iter = 100

# COMMAND ----------

#Helpers for reading the data
def read_data():
    df = spark.read.parquet(TRANSFORMED_TRAINING_DATA)
    training = df.filter(col('YEAR') < 2018)
    testing = df.filter(col('YEAR') == 2018)
    return training, testing


# COMMAND ----------

# MAGIC %md
# MAGIC # TRAIN MODELS

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark ML

# COMMAND ----------

# Load training data
training, testing = read_data()

# COMMAND ----------

display(training.filter(col(label_col).isNull()))
# display(training.select('DEP_DEL15').distinct())

# COMMAND ----------

negatives = training.filter(training[label_col]==0).count()
positives = training.filter(training[label_col]==1).count()
balance_ratio = negatives / (positives+negatives)
training = training.withColumn(class_weights_col, when(training[label_col] == 1, balance_ratio).otherwise(1-balance_ratio))

# COMMAND ----------

display(training)
display(testing)

# COMMAND ----------

# vectorize features
if 'features' in training.columns:
    training = training.drop('features')
training = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(training)

if 'features' in testing.columns:
    testing = testing.drop('features')
testing = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(testing)

# COMMAND ----------

# Train the model and gather training metadata.
lr = LogisticRegression(featuresCol="features", labelCol=label_col, weightCol=class_weights_col, regParam=0.01, elasticNetParam=0.2)
# lr = LogisticRegression(featuresCol=features, labelCol=label, weightCol=class_weights_col)
lrModel = lr.fit(training)
sml_coefficients = ast.literal_eval(str(lrModel.coefficients))
sml_intercept = lrModel.intercept
objectiveHistory = lrModel.summary.objectiveHistory
training_summary = lrModel.summary
train_predictions = lrModel.transform(training)

# COMMAND ----------

lrModel.summary.objectiveHistory

# COMMAND ----------

# Run the model against test dataset, gather results and metrics.
test_predictions = lrModel.transform(testing)
test_metrics = lrModel.evaluate(testing)
test_prediction_result = test_predictions.selectExpr(f'{label_col} as label', 'prediction', 'probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label']), float(row['prediction']))).collect()

# COMMAND ----------

# Print the coefficients and intercept for logistic regression, and save for later
print("Coefficients: " + str(sml_coefficients))
print("Intercept: " + str(sml_intercept))
print(objectiveHistory)

# COMMAND ----------

# traing accuracy
print(f'Training weightedFalsePositiveRate: {training_summary.weightedFalsePositiveRate}')
print(f'Training weightedPrecision: {training_summary.weightedPrecision}')
print(f'Training weightedRecall: {training_summary.weightedRecall}')
print(f'Training weightedTruePositiveRate: {training_summary.weightedTruePositiveRate}')
print(f'Training Accuracy: {training_summary.accuracy}')
print(f'Training F_score (beta=1): {training_summary.weightedFMeasure(1.0)}')
print(f'Training F_beta (beta=0.5): {training_summary.weightedFMeasure(0.5)}')


# COMMAND ----------

# testing accuracy
print(f'Test Accuracy: {test_metrics.accuracy}')
print(f'Test F_score (beta=1): {test_metrics.weightedFMeasure(1.5)}')
print(f'Test F_beta (beta=0.5): {test_metrics.weightedFMeasure(0.5)}')


# COMMAND ----------

display(test_predictions.filter(test_predictions.prediction==1))

# COMMAND ----------

# MAGIC %md ## Evaluate The Model

# COMMAND ----------

# MAGIC %md ### Metrics From Training

# COMMAND ----------

from pyspark.mllib.evaluation import MulticlassMetrics

train_preds_and_labels_rdd = train_predictions.selectExpr(f'{label_col} as label', 'probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
metrics = MulticlassMetrics(train_preds_and_labels_rdd)

print(metrics.confusionMatrix().toArray())

# COMMAND ----------

# MAGIC %md ###Metrics From Testing

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
spark.sparkContext.addPyFile("dbfs:/user/ram.senth@berkeley.edu/curve_metrics.py")
from curve_metrics import CurveMetrics


# Select (prediction, true label) and compute test error
# evaluator = MulticlassClassificationEvaluator(
#     labelCol=label_col, predictionCol='prediction', metricName='accuracy')
# accuracy = evaluator.evaluate(test_predictions)
# print('Test Error = %g' % (1.0 - accuracy))

training_summary.roc.show(5)

test_preds_and_labels_rdd = test_predictions.selectExpr(f'{label_col} as label', 'probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
curvemetrics = CurveMetrics(test_preds_and_labels_rdd)
points = curvemetrics.get_curve('roc')

f = plt.figure()
my_suptitle = f.suptitle('ROC Curve (2019 Data)', fontsize=18, y=1.02) 
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(x_val, y_val)
plt.savefig('/dbfs/user/ram.senth@berkeley.edu/roc.png', dpi=300, bbox_inches='tight',bbox_extra_artists=[my_suptitle])
plt.show()

points = curvemetrics.get_curve('fMeasureByThreshold')
f = plt.figure()
my_suptitle = f.suptitle('fMeasure By Threshold (2019 Data)', fontsize=18, y=1.02) 
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.xlabel('Threshold')
plt.ylabel('F-Measure')
plt.plot(x_val, y_val)
plt.savefig('/dbfs/user/ram.senth@berkeley.edu/fMeasure.png', dpi=300, bbox_inches='tight', bbox_extra_artists=[my_suptitle])
plt.show()


# COMMAND ----------

# Plot the errors/loss in each iteration
f = plt.figure()
my_suptitle = f.suptitle('Loss Curve (2015-2018 Data)', fontsize=18, y=1.02) 
x_val = range(1, len(objectiveHistory) + 1)
y_val = objectiveHistory
# plt.suptitle('Loss Curve (2015-2018 Data)', y=1.02, fontsize=18)
plt.title('(Lower is better)', fontsize=10)
plt.xlabel('Iteration')
plt.ylabel('Loss/Error')
plt.plot(x_val, y_val)
# f.savefig('suptitle_test.pdf', dpi=f.dpi, bbox_inches='tight',bbox_extra_artists=[my_suptitle])
f.savefig('/dbfs/user/ram.senth@berkeley.edu/loss.png', dpi=300, bbox_inches='tight',bbox_extra_artists=[my_suptitle])
plt.show()


# COMMAND ----------

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
import seaborn as sn
from sklearn.metrics import classification_report

# Scikit learn based ROC curve

import sklearn
from sklearn.metrics import fbeta_score
print('The scikit-learn version is {}.'.format(sklearn.__version__))

def report(y_true, y_pred):
    final_report = classification_report(y_true, y_pred)
    print(final_report)
    f_betas = fbeta_score(y_true, y_pred, average=None, beta=0.5)
#     final_report['fBeta(0.5)'] = f_betas
    print(final_report)
    print(f_betas)

def plot_confusion_matrix(df):
    plt.figure(figsize = (10,7))
    sn.heatmap(df[['label', 'prediction']], annot=True)
    plt.show()

def plot_roc(y_test, y_pred):
    logit_roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()    

def scikit_plots():
    df = pd.DataFrame(test_prediction_result, columns =['proba', 'label', 'prediction'])
    print(df.head(5))
    y_test = df['label']
    y_pred = df['prediction']

    plot_roc(y_test, y_pred)
    report(y_test, y_pred)
#     plot_confusion_matrix(df)

scikit_plots()

# COMMAND ----------

# MAGIC %md
# MAGIC Class | Precision | Recall | f1-score | fBeta-score(0.5)
# MAGIC ------|-----------|--------|----------|------------------
# MAGIC 0 | 0.87 | 0.86 | 0.86 | 0.87
# MAGIC 1 | 0.43 | 0.45 | 0.44 | 0.43 | 
# MAGIC 
# MAGIC Table 1: Model Performance (2019 Data)

# COMMAND ----------


