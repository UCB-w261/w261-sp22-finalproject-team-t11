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
from pyspark.sql.functions import col, isnan, when, count, lit, explode, array
from time import *

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

MODEL_PLOTS_BASE_PATH = '/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/model_plots'
TRAINED_MODEL_BASE_PATH = f'{blob_url}/model/trained'

CHECKPOINT_FOLDER = f'{blob_url}/checkpoints'
spark.sparkContext.setCheckpointDir(CHECKPOINT_FOLDER)


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

spark.sparkContext.addPyFile("dbfs:/user/ram.senth@berkeley.edu/curve_metrics.py")
from curve_metrics import CurveMetrics

# import custom cv module
spark.sparkContext.addPyFile("dbfs:/custom_cv.py")
from custom_cv import CustomCrossValidator


# COMMAND ----------

# import custom cv module
spark.sparkContext.addPyFile("dbfs:/custom_cv.py")
from custom_cv import CustomCrossValidator
spark.sparkContext.addPyFile("dbfs:/user/ram.senth@berkeley.edu/fbeta_evaluator.py")
from fbeta_evaluator import FBetaEvaluator

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

training.printSchema()

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
training = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(training).withColumnRenamed(label_col, 'label')
training.checkpoint(eager=True)

if 'features' in testing.columns:
    testing = testing.drop('features')
testing = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(testing).withColumnRenamed(label_col, 'label')
testing.checkpoint(eager=True)

# COMMAND ----------

display(training[feature_cols])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cross Validation Options

# COMMAND ----------

def CV_noCV(df):
    d = {}
    
    d['df1'] = df.filter(df.YEAR <= 2018)\
                       .withColumn('cv', F.when(df.YEAR <= 2017, 'train')
                                             .otherwise('test'))
    return d

def CV_rolling_window(df):
    d = {}

    d['df1'] = df.filter(df.YEAR <= 2016)\
                       .withColumn('cv', F.when(df.YEAR == 2015, 'train')
                                             .otherwise('test'))

    d['df2'] = df.filter(df.YEAR <= 2017)\
                       .withColumn('cv', F.when(df.YEAR <= 2016, 'train')
                                             .otherwise('test'))

    d['df3'] = df.filter(df.YEAR <= 2018)\
                       .withColumn('cv', F.when(df.YEAR <= 2017, 'train')
                                             .otherwise('test'))
    return d

def CV_block_by_quarter(df):
    d = {}
    
    d['df1'] = df.filter((df.YEAR.isin(2015,2016)) & (df.QUARTER == 1))\
                       .withColumn('cv', F.when((df.YEAR == 2015), 'train')
                                             .otherwise('test'))

    d['df2'] = df.filter((df.YEAR.isin(2015,2016)) & (df.QUARTER == 3))\
                       .withColumn('cv', F.when((df.YEAR == 2015), 'train')
                                             .otherwise('test'))

    d['df3'] = df.filter((df.YEAR.isin(2016,2017)) & (df.QUARTER == 2))\
                       .withColumn('cv', F.when((df.YEAR == 2016), 'train')
                                             .otherwise('test'))

    d['df4'] = df.filter((df.YEAR.isin(2016,2017)) & (df.QUARTER == 4))\
                       .withColumn('cv', F.when((df.YEAR == 2016), 'train')
                                             .otherwise('test'))

    d['df5'] = df.filter((df.YEAR.isin(2017,2018)) & (df.QUARTER == 1))\
                       .withColumn('cv', F.when((df.YEAR == 2017), 'train')
                                             .otherwise('test'))

    d['df6'] = df.filter((df.YEAR.isin(2017,2018)) & (df.QUARTER == 3))\
                       .withColumn('cv', F.when((df.YEAR == 2017), 'train')
                                             .otherwise('test'))

    d['df7'] = df.filter((df.YEAR.isin(2015,2018)) & (df.QUARTER == 2))\
                       .withColumn('cv', F.when((df.YEAR == 2015), 'train')
                                             .otherwise('test'))

    d['df8'] = df.filter((df.YEAR.isin(2015,2018)) & (df.QUARTER == 4))\
                       .withColumn('cv', F.when((df.YEAR == 2015), 'train')
                                             .otherwise('test'))
    return d

# COMMAND ----------

# MAGIC %md
# MAGIC ## Addressing Imbalanced Classes

# COMMAND ----------

def address_imbalance(df, weights_bool):
    
    def add_class_weights(df):
        label_col = 'label'
        class_weights_col = 'classWeights'
        negatives = df.filter(df[label_col]==0).count()
        positives = df.filter(df[label_col]==1).count()
        balance_ratio = negatives / (positives+negatives)
        return df.withColumn(class_weights_col, when(training[label_col] == 1, balance_ratio).otherwise(1-balance_ratio))

    def oversample_minority_class(df):
        label_col = 'label'
        minor_df = df.where(df[label_col] == 1)
        major_df = df.where(df[label_col] == 0)
        # ratio = int(negatives/positives) # defined in source
        ratio = 3 # for experimentation, can adjust
        minor_oversampled = minor_df.withColumn('dummy', explode(array([lit(x) for x in range(ratio)]))).drop('dummy')
        return major_df.unionAll(minor_oversampled)
      
    def undersample_minority_class(df):
        label_col = 'label'
        minor_df = df.where(df[label_col] == 1)
        major_df = df.where(df[label_col] == 0)
        # ratio = int(negatives/positives) # defined in source
        ratio = 3 # for experimentation, can adjust
        major_undersampled = major_df.sample(False, 1/ratio)
        all_undersampled = major_undersampled.unionAll(minor_df)
#         all_undersampled.checkpoint(eager=True)
        return all_undersampled
        

    if (weights_bool == True):
        return add_class_weights(df)
    else:
        return undersample_minority_class(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest

# COMMAND ----------

# import custom cv module
spark.sparkContext.addPyFile("dbfs:/custom_cv.py")
from custom_cv import CustomCrossValidator
spark.sparkContext.addPyFile("dbfs:/user/ram.senth@berkeley.edu/fbeta_evaluator.py")
from fbeta_evaluator import FBetaEvaluator

# COMMAND ----------

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
d = CV_block_by_quarter()
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

training_summary = cvModel.summary



# COMMAND ----------

best = cvModel.bestModel
def ExtractFeatureCoeficient(model, dataset, excludedCols = None):
    test = model.transform(dataset)
    weights = model.coefficients
    print('This is model weights: \n', weights)
    weights = [(float(w),) for w in weights]  # convert numpy type to float, and to tuple
    if excludedCols == None:
        feature_col = [f for f in test.schema.names if f not in ['y', 'classWeights', 'features', 'label', 'rawPrediction', 'probability', 'prediction']]
    else:
        feature_col = [f for f in test.schema.names if f not in excludedCols]
    if len(weights) == len(feature_col):
        weightsDF = sqlContext.createDataFrame(zip(weights, feature_col), schema= ["Coeficients", "FeatureName"])
    else:
        print('Coeficients are not matching with remaining Fetures in the model, please check field lists with model.transform(dataset).schema.names')
    
    return weightsDF

best = cvModel.bestModel
results = ExtractFeatureCoeficient(best, training)
results.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Try 2

# COMMAND ----------

# vectorize features
feature_cols = ['origin_weather_Avg_Elevation',
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

# if 'features' in training.columns:
#     training = training.drop('features')
# training = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(training).withColumnRenamed(label_col, 'label').cache()

# if 'features' in testing.columns:
#     testing = testing.drop('features')
# testing = VectorAssembler(inputCols=feature_cols, outputCol="features").transform(testing).withColumnRenamed(label_col, 'label').cache()

# COMMAND ----------

training.printSchema()

# COMMAND ----------

def prepare_data(df, feature_cols, label_col, weights_bool):
    label_col = 'DEP_DEL15'
    if 'features' in df.columns:
        df = df.drop('features')
    df = VectorAssembler(inputCols=feature_cols, 
                             outputCol="features").transform(df).withColumnRenamed(label_col, 'label')
    df = address_imbalance(df, weights_bool)
    return df
    
weights_bool = False
training_use = prepare_data(training, feature_cols, label_col, weights_bool)

# COMMAND ----------

training_use.printSchema()

# COMMAND ----------

def save_model(model, model_name):
    model.write().overwrite().save(f'{TRAINED_MODEL_BASE_PATH}/{model_name}')

# COMMAND ----------

# set up grid search: estimator, set of params, and evaluator

if weights_bool:
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", weights = 'classWeights')
else:
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
grid = ParamGridBuilder()\
            .addGrid(rf.maxDepth, [5, 20])\
            .addGrid(rf.numTrees, [10, 30])\
            .build()
# evaluator = FBetaEvaluator()
evaluator = BinaryClassificationEvaluator()

# run cross validation
cv = CustomCrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator,
     splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
# d = CV_rolling_window(training_use)
d = CV_noCV(training_use)
cvModel = cv.fit(d)
save_model(cvModel, 'rf_baseline')

# COMMAND ----------

# make predictions
predictions = cvModel.transform(training_use)
display(predictions.groupby('label', 'prediction').count())

# COMMAND ----------

best = cvModel.bestModel

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))

def extract_feature_importance(model, predictions):
    training_summary = model.summary
    importance = ExtractFeatureImp(cvModel.bestModel.featureImportances, predictions, "features").head(25)
    importance.sort_values(by=["score"], inplace = True, ascending=False)
    display(importance)

# best = cvModel.bestModel
extract_feature_importance(best, predictions)

# COMMAND ----------

def print_training_summary(model):
    summary = model.summary
    print(f'Training weightedFalsePositiveRate: {summary.weightedFalsePositiveRate}')
    print(f'Training weightedPrecision: {summary.weightedPrecision}')
    print(f'Training weightedRecall: {summary.weightedRecall}')
    print(f'Training weightedTruePositiveRate: {summary.weightedTruePositiveRate}')
    print(f'Training Accuracy: {summary.accuracy}')
    print(f'Training F_score (beta=1): {summary.weightedFMeasure(1.0)}')
    print(f'Training F_beta (beta=0.5): {summary.weightedFMeasure(0.5)}')
    
print_training_summary(cvModel.bestModel)

# COMMAND ----------

cvModel.bestModel.getMaxDepth()

# COMMAND ----------

model_summary = [['Location', f'{TRAINED_MODEL_BASE_PATH}/{MODEL_NAME}'],
    ['Features', cvModel.bestModel.numFeatures], \
    ['Max Depth', cvModel.bestModel.getMaxDepth()], \
    ['Num Trees', cvModel.bestModel.getNumTrees], \
    ['Class Imbalance Strategy', 'Undersample Majority Class'], \
    ['Cross Validation Strategy', 'None (2015-2017 - training; 2018 - validation)'],\
    ['Time To Train', '14 minutes'],
    ['2019-TP', 1699505]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Consolidate Everything

# COMMAND ----------

def save_model(model, model_name, imbalance_strategy, cv_strategy, train_time):
    """
        Save the model itself and summary stats as csv
    """
    model.write().overwrite().save(f'{TRAINED_MODEL_BASE_PATH}/{model_name}')
    model_summary = [('Location', f'{TRAINED_MODEL_BASE_PATH}/{model_name}'),
                     ('Features', str(model.bestModel.numFeatures)), \
                     ('Max Depth', str(model.bestModel.getMaxDepth())), \
                     ('Num Trees', str(model.bestModel.getNumTrees)), \
                     ('Class Imbalance Strategy', imbalance_strategy), \
                     ('Cross Validation Strategy', cv_strategy),\
                     ('Time To Train', train_time)]
    summary_cols = ['Stat', 'Model Value']
    model_summary_df = spark.createDataFrame(data = model_summary, schema = summary_cols)
    model_summary_df.write.mode('overwrite').csv(f'{TRAINED_MODEL_BASE_PATH}/{model_name}')
    
def address_imbalance(df, weights_bool):
    """
        Either add class weights or undersample the minority class
    """
    
    def add_class_weights(df):
        label_col = 'label'
        class_weights_col = 'classWeights'
        negatives = df.filter(df[label_col]==0).count()
        positives = df.filter(df[label_col]==1).count()
        balance_ratio = negatives / (positives+negatives)
        return df.withColumn(class_weights_col, when(training[label_col] == 1, balance_ratio).otherwise(1-balance_ratio))

    def oversample_minority_class(df):
        label_col = 'label'
        minor_df = df.where(df[label_col] == 1)
        major_df = df.where(df[label_col] == 0)
        # ratio = int(negatives/positives) # defined in source
        ratio = 3 # for experimentation, can adjust
        minor_oversampled = minor_df.withColumn('dummy', explode(array([lit(x) for x in range(ratio)]))).drop('dummy')
        return major_df.unionAll(minor_oversampled)
      
    def undersample_minority_class(df):
        label_col = 'label'
        minor_df = df.where(df[label_col] == 1)
        major_df = df.where(df[label_col] == 0)
        # ratio = int(negatives/positives) # defined in source
        ratio = 3 # for experimentation, can adjust
        major_undersampled = major_df.sample(False, 1/ratio)
        all_undersampled = major_undersampled.unionAll(minor_df)
#         all_undersampled.checkpoint(eager=True)
        return all_undersampled
    
    if (weights_bool == True):
        return add_class_weights(df)
    else:
        return undersample_minority_class(df)
    
def prepare_data(df, train_bool, feature_cols, label_col, weights_bool):
    """
        Prepare data for modeling by applying a Vector Assembler and addressing class imbalance
    """
    label_col = 'DEP_DEL15'
    if 'features' in df.columns:
        df = df.drop('features')
    df = VectorAssembler(inputCols=feature_cols, 
                             outputCol="features").transform(df).withColumnRenamed(label_col, 'label')
    if train_bool:
        df = address_imbalance(df, weights_bool)
    return df

def run_random_forest(label_col, class_weights_col, params_maxDepth, params_numTrees, evaluator, dataset, model_name, cv_strategy = 'None', features_col = 'features'):
    """
        Runs a random forest regression model with given parameters.
        maxDepth and numTrees are tuned using grid search
    """
    
    if class_weights_col is None:
        rf = RandomForestClassifier(labelCol = label_col, featuresCol = features_col)
        imbalance_strategy = 'Undersample Majority Class'
    else:
        rf = RandomForestClassifier(labelCol = label_col, featuresCol = features_col, weights = class_weights_col)
        imbalance_strategy = 'Class Weights'
        
    grid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, params_maxDepth)\
        .addGrid(rf.numTrees, params_numTrees)\
        .build()

    start_time = time()
    
    # run cross validation
    cv = CustomCrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator,
                              splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)

    cvModel = cv.fit(dataset)
    
    end_time = time()
    train_time = f'{round((end_time - start_time)/60, 2)} minutes'
    
    save_model(cvModel, model_name, imbalance_strategy, cv_strategy, train_time)
    
    return cvModel

def print_predictions_breakdown(model, dataset):
    """
        Prints out the confusion matrix for a given dataset
    """
    predictions = model.transform(dataset)
    display(predictions.groupby(label_col, 'prediction').count())
    
def print_training_summary(model):
    """
        Prints out model performance on training/validation dataset.
    """
    print('Metrics from model.avgMetrics')
    print (f'Average F1-score: {model.avgMetrics[0]}')
    print (f'Average Weighted Precision: {model.avgMetrics[1]}')
    print (f'Average Weighted Recall: {model.avgMetrics[2]}')
    print (f'Average Accuracy: {model.avgMetrics[3]}')

    summary = model.bestModel.summary
    print('Metrics from model summary')
    print(f'Training weightedFalsePositiveRate: {summary.weightedFalsePositiveRate}')
    print(f'Training weightedPrecision: {summary.weightedPrecision}')
    print(f'Training weightedRecall: {summary.weightedRecall}')
    print(f'Training weightedTruePositiveRate: {summary.weightedTruePositiveRate}')
    print(f'Training Accuracy: {summary.accuracy}')
    print(f'Training F_score (beta=1): {summary.weightedFMeasure(1.0)}')
    print(f'Training F_beta (beta=0.5): {summary.weightedFMeasure(0.5)}')
    


# COMMAND ----------

# vectorize features
feature_cols = ['origin_weather_Avg_Elevation',
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

# COMMAND ----------

# Load training data and test data
training = spark.read.parquet(TRANSFORMED_TRAINING_DATA)
# testing = spark.read.parquet(TRANSFORMED_2019_DATA)

label_col = 'DEP_DEL15'
weights_bool = False
cv_strategy = 'None'
evaluator = BinaryClassificationEvaluator()

# Apply data transformations needed for training and testing. Apply undersampling to training set.
training_use = prepare_data(df = training, train_bool = True, feature_cols = feature_cols, label_col = label_col, 
                            weights_bool = weights_bool)
testing_use = prepare_data(testing, False, feature_cols, label_col, weights_bool)


# Check point the data for optimization.
training_use.checkpoint(eager=True)

# Apply cross validation strategy and prepare the training dataset with required folds.
if cv_strategy == 'None':
    training_withsplits = CV_noCV(training_use)
elif cv_strategy == 'Rolling Window':
    training_withsplits = CV_rolling_window(training_use)
else:
    training_withsplits = CV_block_by_quarter(training_use)

# Load test data
# testing_2019 = spark.read.parquet(TRANSFORMED_2019_DATA)

# Apply data transformations needed for using the model.
# testing_2019 = prepare_data(testing_2019, False, feature_cols, label_col, weights_bool)

# COMMAND ----------

rf_model = run_random_forest(label_col = 'label', class_weights_col = None, params_maxDepth = [20], params_numTrees = [30], evaluator = evaluator, dataset = training_withsplits, model_name = 'rf_baseline_test', cv_strategy = cv_strategy, features_col = 'features')

# COMMAND ----------


weights_bool = True
training_use = prepare_data(training, feature_cols, label_col, weights_bool).cache()


# set up grid search: estimator, set of params, and evaluator

if weights_bool:
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", weights = 'classWeights')
else:
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
grid = ParamGridBuilder()\
            .addGrid(rf.maxDepth, [10, 15, 20, 25, 30, 35])\
            .addGrid(rf.numTrees, [10, 15, 20, 25, 30, 35, 40, 45, 50])\
            .build()
# evaluator = FBetaEvaluator()
evaluator = BinaryClassificationEvaluator()

# run cross validation
cv = CustomCrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator,
     splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
d = CV_rolling_window(training_use)
cvModel = cv.fit(d)

# make predictions
predictions = cvModel.transform(training)
display(predictions.groupby('label', 'prediction').count())

best = cvModel.bestModel
extract_feature_importance(best, predictions)

print_training_summary(best)

# COMMAND ----------


