# Databricks notebook source
# MAGIC %md
# MAGIC # Logistic Regression - Model Search with Cross Validation
# MAGIC 
# MAGIC In this notebook we implement the search of an optimal logistic regression model, leveraging a customized verion of the CrossValidation algorithm to take into account the time series nature of our data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook setup

# COMMAND ----------

# general imports
import pandas as pd
import matplotlib
import time
import matplotlib.pyplot as plt
import ast

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, LogisticRegression
from pyspark.ml.tuning import CrossValidatorModel

from custom_cv import CustomCrossValidator
from configuration_v01 import Configuration
from data_preparer_v06 import DataPreparer, CVSplitter, CVStrategies
from training_summarizer_v01 import TrainingSummarizer
configuration = Configuration()
data_preparer = DataPreparer(configuration)
cvStrategies = CVStrategies()
training_summarizer = TrainingSummarizer(configuration)

# magic commands
%matplotlib inline
%reload_ext autoreload
%autoreload 2

# COMMAND ----------

configuration.feature_cols

# COMMAND ----------

# Notebook specific constants
# Full dataset
dataset_to_use = configuration.FULL_DATASET
MODEL_NAME = 'lr_gridsearch'
class_imbalance_strategy = 'Undersample Majority Class'
cross_validation_strategy = cvStrategies.block_years
params_reg = [0.01, 0.025, 0.05, 0.1]
params_elastic_net = [0.01, 0.1, 0.25]
params_max_iter = [30, 50, 100]

# dataset_to_use = configuration.FULL_DATASET
# MODEL_NAME = 'lr_baseline'
# class_imbalance_strategy = 'Undersample Majority Class'
# cross_validation_strategy = cvStrategies.no_cv
# params_reg = [0.01]
# params_elastic_net = [0.01]
# params_max_iter = [30]

# Toy dataset
# dataset_to_use = configuration.TOY_DATASET
# MODEL_NAME = 'lr_toy'
# class_imbalance_strategy = 'Undersample Majority Class'
# cross_validation_strategy = cvStrategies.no_cv
# params_reg = [0.01]
# params_elastic_net = [0.01]
# params_max_iter = [30]


# COMMAND ----------

# MAGIC %md ## Load And Prepare Data
# MAGIC 
# MAGIC Here, we
# MAGIC * load testing and 
# MAGIC   * use vector assembler to assemble the feature vectors
# MAGIC   * under sample the majority class to address class imbalance
# MAGIC   * create folds for cross validation
# MAGIC * load test data and use vector assembler to assemble the feature vectors

# COMMAND ----------

training_withsplits, training, testing = data_preparer.load_and_prep_data(spark, dataset_to_use,\
                                                                          cross_validation_strategy,
                                                                          apply_class_weights=False)

# COMMAND ----------

# MAGIC %md ## Model Training

# COMMAND ----------

def run_logistic_regression(label_col, class_weights_col, features_col, \
                            params_reg, params_elastic_net, params_max_iter, evaluator, dataset):
    """
        Runs a logistic regression with given parameters. L1, L2 regularization parameters as well as the max iteration parameters are
        tuned using grid search.
    """
    if class_weights_col is None:
        lr = LogisticRegression(labelCol=label_col, featuresCol=features_col)
    else:
        lr = LogisticRegression(labelCol=label_col, weightCol=class_weights_col, featuresCol=features_col)
    grid = ParamGridBuilder()\
                .baseOn({lr.labelCol: label_col}) \
                .baseOn([lr.predictionCol, 'prediction']) \
                .addGrid(lr.regParam, params_reg) \
                .addGrid(lr.elasticNetParam, params_elastic_net) \
                .addGrid(lr.maxIter, params_max_iter) \
                .build()
    cv = CustomCrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator,
          splitWord = ('train', 'test'), cvCol = 'cv', parallelism=4)
    cv_model = cv.fit(dataset)
    return cv_model


# COMMAND ----------

# class_weights_col is None as we are using undersampling of dominant class.
start = time.time()

cvModel = run_logistic_regression(label_col=configuration.label_col, class_weights_col=None, \
                                  features_col=configuration.features_col, dataset=training_withsplits, \
                                  params_reg=params_reg, \
                                  params_elastic_net=params_elastic_net, \
                                  params_max_iter=params_max_iter, \
                                  evaluator=BinaryClassificationEvaluator())
# Save the model
cvModel.write().overwrite().save(f'{configuration.TRAINED_MODEL_BASE_PATH}/{MODEL_NAME}')

training_time_seconds = time.time() - start

# COMMAND ----------

# MAGIC %md ## Summarize Model Performance

# COMMAND ----------

    
# def plot_coefficients(model):
#     sml_coefficients = ast.literal_eval(str(model.bestModel.coefficients))
#     coefs = pd.DataFrame(data = {'col_names' : configuration.feature_cols[0:5], 'coef_vals' : sml_coefficients[0:5]}).sort_values(by = 'coef_vals', ascending = True)
#     plt.barh(y = coefs['col_names'], width = coefs['coef_vals'])
#     plt.title('Top 5 Coefficients in Baseline Model')
#     plt.show()
#     plt.savefig('/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/model_plots/lr_coefficients.png', dpi=300, bbox_inches='tight'); 

def plot_coefficients(model, training_dataset):
    lr_features = [x["name"] for x in sorted(training_dataset.schema[configuration.features_col].metadata["ml_attr"]["attrs"]["binary"] + \
       training_dataset.schema[configuration.features_col].metadata["ml_attr"]["attrs"]["numeric"], \
       key=lambda x: x["idx"])]

    sml_coefficients = ast.literal_eval(str(model.bestModel.coefficients))

    coefs = pd.DataFrame(
        data = {'col_id': list(range(1, len(sml_coefficients) + 1)), 'col_name': lr_features, \
                'coef_vals': sml_coefficients})
    coefs['coef_abs'] = coefs['coef_vals'].abs()

    # Save the coefficients
    # coefs.write().overwrite().save(f'{configuration.TRAINED_MODEL_BASE_PATH}/{MODEL_NAME}_coeffs')
    spark.createDataFrame(coefs).write.mode('overwrite').csv(f'{configuration.TRAINED_MODEL_BASE_PATH}/{MODEL_NAME}_coeffs')

    # Sort desc to get the top 30 and then sort asc as matplot needs it flipped.
    coefs = coefs.sort_values(by = 'coef_abs', ascending = False)[0:30]
    coefs = coefs.sort_values(by = 'coef_abs', ascending = True)
    
    fig = plt.figure(figsize=(12, 8))
    plt.barh(y = coefs['col_name'], width = coefs['coef_vals'])
    plt.xticks(rotation=90)
    plt.title(f'Top 30 Coefficients ({MODEL_NAME})')
    plt.show()
    plt.savefig(f'{configuration.MODEL_PLOTS_BASE_PATH}/{MODEL_NAME}_coefficients_hires.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{configuration.MODEL_PLOTS_BASE_PATH}/{MODEL_NAME}_coefficients_lowres.png', bbox_inches='tight')

# COMMAND ----------

model_summary = training_summarizer.get_summary(cvModel, 'Logistic Regression', MODEL_NAME, \
                    dataset_to_use, training_time_seconds, class_imbalance_strategy, \
                    cross_validation_strategy, training, testing)
training_summarizer.save_model_summary(spark, sc, dbutils, model_summary, overwrite=False)

# COMMAND ----------

training_summarizer.plot_loss_curve(cvModel.bestModel.summary.objectiveHistory, \
                MODEL_NAME,
                dataset_to_use.training_dataset_name,
                f'{configuration.MODEL_PLOTS_BASE_PATH}/{MODEL_NAME}_loss_hires.png', \
                f'{configuration.MODEL_PLOTS_BASE_PATH}/{MODEL_NAME}_loss_lowres.png')

# COMMAND ----------

training_summarizer.test_plots(cvModel, training, MODEL_NAME, \
                               dataset_to_use.test_dataset_name, \
                               f'{configuration.MODEL_PLOTS_BASE_PATH}/{MODEL_NAME}_roc', \
                               f'{configuration.MODEL_PLOTS_BASE_PATH}/{MODEL_NAME}_fMeasure')

# COMMAND ----------

plot_coefficients(cvModel, training)

# COMMAND ----------

def print_summary(location):
    df = spark.read.parquet(location)
    display(df)
    df.printSchema()

print_summary(configuration.MODEL_HISTORY_PATH)

# COMMAND ----------


