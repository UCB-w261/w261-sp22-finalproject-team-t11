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
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import ast
from time import *
from datetime import datetime

from pyspark.ml.tuning import ParamGridBuilder

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# magic commands
%matplotlib inline
%reload_ext autoreload
%autoreload 2

from custom_cv import CustomCrossValidator
from configuration_v01 import Configuration
from data_preparer_v05 import DataPreparer, CVSplitter, CVStrategies
from training_summarizer_v01 import TrainingSummarizer
configuration = Configuration()
data_preparer = DataPreparer(configuration)
cvStrategies = CVStrategies()
training_summarizer = TrainingSummarizer(configuration)

# COMMAND ----------

def ExtractFeatureImp(featureImp, dataset):
    list_extract = []
    for i in dataset.schema[configuration.features_col].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[configuration.features_col].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))

def extract_feature_importance(cvModel, predictions):
    model = cvModel.bestModel
    training_summary = model.summary
    importance = ExtractFeatureImp(cvModel.bestModel.featureImportances, predictions).head(25)
    importance.sort_values(by=["score"], inplace = True, ascending=False)
    display(importance)


# COMMAND ----------

def run_random_forest(dataset, model_name, apply_class_weights, params_maxDepth, params_numTrees, evaluator, cv_strategy='None'):
    """
        Runs a random forest regression model with given parameters.
        maxDepth and numTrees are tuned using grid search
    """

    # Load data
    training_withsplits, training, testing = data_preparer.load_and_prep_data(spark, dataset, cv_strategy, apply_class_weights)

    # Create the model to train.
    if apply_class_weights:
        rf = RandomForestClassifier(labelCol=configuration.label_col, featuresCol=configuration.features_col, weights=class_weights_col)
        class_imbalance_strategy = 'Class Weights'
    else:
        rf = RandomForestClassifier(labelCol=configuration.label_col, featuresCol=configuration.features_col)
        class_imbalance_strategy = 'Undersample Majority Class'
        
    grid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, params_maxDepth)\
        .addGrid(rf.numTrees, params_numTrees)\
        .build()

    # Train the model
    start_time = time()
    cv = CustomCrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator,
                              splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
    cvModel = cv.fit(training_withsplits)
    end_time = time()
    train_time_seconds = end_time - start_time
    
    # Save trained model
    cvModel.write().overwrite().save(f'{configuration.TRAINED_MODEL_BASE_PATH}/{model_name}')
    
    # Analyze and save results.
    model_summary = training_summarizer.get_summary(cvModel, 'Random Forest', model_name, \
                    dataset, train_time_seconds, class_imbalance_strategy, \
                    cv_strategy, training, testing)

    training_summarizer.save_model_summary(spark, sc, dbutils, model_summary, overwrite=False)
    return cvModel, training, testing


# COMMAND ----------

# MAGIC %md
# MAGIC ## Baseline Random Forest

# COMMAND ----------

# Baseline using toy dataset
cvModel, training, testing = run_random_forest(dataset=configuration.TOY_DATASET, model_name='rf_toy', \
                            apply_class_weights=False, params_maxDepth=[20], params_numTrees=[30], 
                            evaluator=BinaryClassificationEvaluator(), cv_strategy=cvStrategies.no_cv)

# COMMAND ----------

# Baseline using full dataset
cvModel, training, testing = run_random_forest(dataset=configuration.FULL_DATASET, model_name='rf_baseline', \
                            apply_class_weights=False, params_maxDepth=[20], params_numTrees=[30], 
                            evaluator=BinaryClassificationEvaluator(), cv_strategy=cvStrategies.no_cv)


# COMMAND ----------

predictions = cvModel.transform(training)

extract_feature_importance(cvModel, predictions)

predictions = cvModel.transform(training)

extract_feature_importance(cvModel, predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grid Search
# MAGIC 
# MAGIC We haven't been successful with running Random Forest grid search. The following command is hence commented out.

# COMMAND ----------

# cvModel, training, testing = run_random_forest(dataset=configuration.FULL_DATASET, model_name='rf_gridsearch', \
#                                                apply_class_weights=False, params_maxDepth=[10, 15, 25], \
#                                                params_numTrees=[10, 20, 30], \
#                                                evaluator=BinaryClassificationEvaluator(), cv_strategy=cvStrategies.block_years)

# # make predictions
# predictions = cvModel.transform(training)
# extract_feature_importance(cvModel, predictions)
