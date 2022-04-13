# Databricks notebook source
# MAGIC %md # TrainingSummarizer

# COMMAND ----------

filename = 'training_summarizer_v01.py'
from configuration_v01 import Configuration
configuration = Configuration()


# COMMAND ----------

# MAGIC %%writefile /dbfs/user/ram.senth@berkeley.edu/tmp/training_summarizer_v01.py
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC import json
# MAGIC from pyspark.ml.tuning import CrossValidatorModel
# MAGIC from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
# MAGIC import matplotlib
# MAGIC import matplotlib.pyplot as plt
# MAGIC from curve_metrics import CurveMetrics
# MAGIC from sklearn.metrics import classification_report, accuracy_score, fbeta_score
# MAGIC from pyspark.sql.functions import current_timestamp
# MAGIC from utility_01 import Utility
# MAGIC 
# MAGIC class TrainingSummarizer():
# MAGIC     def version():
# MAGIC         return '1.0'
# MAGIC 
# MAGIC     def __init__(self, configuration):
# MAGIC         self.configuration = configuration
# MAGIC 
# MAGIC     def load_saved_model(self, model_name):
# MAGIC         return CrossValidatorModel.read().load(f'{self.configuration.TRAINED_MODEL_BASE_PATH}/{model_name}')
# MAGIC     
# MAGIC     def print_training_summary(self, model):
# MAGIC         """
# MAGIC             Prints out model performance on training/validation dataset.
# MAGIC         """
# MAGIC         print('Metrics from model.avgMetrics')
# MAGIC         print (f'Average F1-score: {model.avgMetrics[0]}')
# MAGIC         print (f'Average Weighted Precision: {model.avgMetrics[1]}')
# MAGIC         print (f'Average Weighted Recall: {model.avgMetrics[2]}')
# MAGIC         print (f'Average Accuracy: {model.avgMetrics[3]}')
# MAGIC 
# MAGIC         summary = model.bestModel.summary
# MAGIC         print('Metrics from model summary')
# MAGIC         print(f'Training weightedFalsePositiveRate: {summary.weightedFalsePositiveRate}')
# MAGIC         print(f'Training weightedPrecision: {summary.weightedPrecision}')
# MAGIC         print(f'Training weightedRecall: {summary.weightedRecall}')
# MAGIC         print(f'Training weightedTruePositiveRate: {summary.weightedTruePositiveRate}')
# MAGIC         print(f'Training Accuracy: {summary.accuracy}')
# MAGIC         print(f'Training F_score (beta=1): {summary.weightedFMeasure(1.0)}')
# MAGIC         print(f'Training F_beta (beta=0.5): {summary.weightedFMeasure(0.5)}')
# MAGIC 
# MAGIC     def plot_loss_curve(self, objective_history, model_name, data_name, file_to_save_hires=None, file_to_save_lowres=None):
# MAGIC         """
# MAGIC             Plots the loss curve during model training. 
# MAGIC         """
# MAGIC         f = plt.figure()
# MAGIC         my_suptitle = f.suptitle(f'{model_name} - Loss Curve ({data_name})', fontsize=18, y=1.02) 
# MAGIC         x_val = range(1, len(objective_history) + 1)
# MAGIC         y_val = objective_history
# MAGIC         # plt.suptitle('Loss Curve (2015-2018 Data)', y=1.02, fontsize=18)
# MAGIC         plt.title('(Lower is better)', fontsize=10)
# MAGIC         plt.xlabel('Iteration')
# MAGIC         plt.ylabel('Loss/Error')
# MAGIC         plt.plot(x_val, y_val)
# MAGIC         if file_to_save_hires is not None:
# MAGIC             f.savefig(file_to_save_hires, dpi=300, bbox_inches='tight', bbox_extra_artists=[my_suptitle])
# MAGIC         if file_to_save_lowres is not None:
# MAGIC             f.savefig(file_to_save_lowres, bbox_inches='tight', bbox_extra_artists=[my_suptitle])
# MAGIC         plt.show()    
# MAGIC 
# MAGIC     def test_plots(self, model, dataset, model_name, data_name, roc_file=None, fmeasure_by_threshold_file=None):
# MAGIC         test_predictions = model.bestModel.transform(dataset)
# MAGIC         test_metrics = model.bestModel.evaluate(dataset)
# MAGIC 
# MAGIC         test_preds_and_labels_rdd = test_predictions.selectExpr(self.configuration.label_col, 'probability').rdd.map(lambda row: (float(row['probability'][1]), float(row[self.configuration.label_col])))
# MAGIC         curvemetrics = CurveMetrics(test_preds_and_labels_rdd)
# MAGIC         points = curvemetrics.get_curve('roc')
# MAGIC 
# MAGIC         f = plt.figure()
# MAGIC         my_suptitle = f.suptitle(f'{model_name} - ROC Curve ({data_name})', fontsize=18, y=1.02) 
# MAGIC         x_val = [x[0] for x in points]
# MAGIC         y_val = [x[1] for x in points]
# MAGIC         plt.xlabel('False Positive Rate')
# MAGIC         plt.ylabel('True Positive Rate')
# MAGIC         plt.plot(x_val, y_val)
# MAGIC 
# MAGIC         if roc_file is not None:
# MAGIC             plt.savefig(f'{roc_file}_hires.png', dpi=300, bbox_inches='tight', bbox_extra_artists=[my_suptitle])
# MAGIC             plt.savefig(f'{roc_file}_lores.png', bbox_inches='tight', bbox_extra_artists=[my_suptitle])
# MAGIC         plt.show()
# MAGIC 
# MAGIC         points = curvemetrics.get_curve('fMeasureByThreshold')
# MAGIC         f = plt.figure()
# MAGIC         my_suptitle = f.suptitle(f'{model_name} - fMeasure By Threshold ({data_name})', fontsize=18, y=1.02) 
# MAGIC         x_val = [x[0] for x in points]
# MAGIC         y_val = [x[1] for x in points]
# MAGIC         plt.xlabel('Threshold')
# MAGIC         plt.ylabel('F-Measure')
# MAGIC         plt.plot(x_val, y_val)
# MAGIC 
# MAGIC         if fmeasure_by_threshold_file is not None:
# MAGIC             plt.savefig(f'{fmeasure_by_threshold_file}_hires.png', dpi=300, bbox_inches='tight', bbox_extra_artists=[my_suptitle])
# MAGIC             plt.savefig(f'{fmeasure_by_threshold_file}_lowres.png', bbox_inches='tight', bbox_extra_artists=[my_suptitle])
# MAGIC         plt.show()
# MAGIC 
# MAGIC #     def report(self, y_true, y_pred):
# MAGIC #         final_report = classification_report(y_true, y_pred)
# MAGIC #         print(final_report)
# MAGIC #         f_betas = fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
# MAGIC #     #     final_report['fBeta(0.5)'] = f_betas
# MAGIC #         print(final_report)
# MAGIC #         print(f'F-Beta(0.5): {f_betas}')
# MAGIC 
# MAGIC #     def scikit_plots(self, model, dataset):
# MAGIC #         test_prediction = model.transform(dataset)
# MAGIC #         preds = test_prediction.select(['probability', self.configuration.label_col, 'prediction']).collect()
# MAGIC #         df = pd.DataFrame(preds, columns =['probability', self.configuration.label_col, 'prediction'])
# MAGIC #         y_test = df[self.configuration.label_col]
# MAGIC #         y_pred = df['prediction']
# MAGIC #         report(y_test, y_pred)
# MAGIC #     #     plot_roc(y_test, y_pred)
# MAGIC #     #     plot_confusion_matrix(df)
# MAGIC 
# MAGIC     def get_predictions_summary(self, predictions):
# MAGIC         grouped = predictions.groupby(self.configuration.label_col, 'prediction').count().toPandas()
# MAGIC         return {
# MAGIC             'TP': grouped[(grouped['label'] == 1) & (grouped['prediction'] == 1)].iat[0, 2],
# MAGIC             'TN': grouped[(grouped['label'] == 0) & (grouped['prediction'] == 0)].iat[0, 2],
# MAGIC             'FP': grouped[(grouped['label'] == 0) & (grouped['prediction'] == 1)].iat[0, 2],
# MAGIC             'FN': grouped[(grouped['label'] == 1) & (grouped['prediction'] == 0)].iat[0, 2]
# MAGIC         }
# MAGIC 
# MAGIC     def get_training_scores(self, model):
# MAGIC         summary = model.bestModel.summary
# MAGIC         return {
# MAGIC             'weightedFalsePositiveRate': summary.weightedFalsePositiveRate,
# MAGIC             'weightedPrecision': summary.weightedPrecision,
# MAGIC             'weightedRecall': summary.weightedRecall,
# MAGIC             'weightedTruePositiveRate': summary.weightedTruePositiveRate,
# MAGIC             'Accuracy': summary.accuracy,
# MAGIC             'F_score (beta=1)': summary.weightedFMeasure(1.0),
# MAGIC             'F_beta (beta=0.5)': summary.weightedFMeasure(0.5)
# MAGIC         }
# MAGIC 
# MAGIC     def get_scores(self, predictions):
# MAGIC         preds = predictions.select(['probability', self.configuration.label_col, 'prediction']).collect()
# MAGIC         df = pd.DataFrame(preds, columns =['probability', self.configuration.label_col, 'prediction'])
# MAGIC         y_true = df[self.configuration.label_col]
# MAGIC         y_pred = df['prediction']
# MAGIC         final_report = classification_report(y_true, y_pred, output_dict=True)
# MAGIC         f_beta = fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
# MAGIC         weighted_accuracy = accuracy_score(y_true, y_pred)    
# MAGIC 
# MAGIC         return {
# MAGIC             'Accuracy': weighted_accuracy,
# MAGIC             'Precision': final_report.get('weighted avg').get('precision'),
# MAGIC             'Recall': final_report.get('weighted avg').get('recall'),
# MAGIC             'F_score (beta=1)': final_report.get('weighted avg').get('f1-score'),
# MAGIC             'F_beta (beta=0.5)': f_beta
# MAGIC         }
# MAGIC 
# MAGIC     def save_model_summary(self, spark, sc, dbutils, model_summary, overwrite=False):
# MAGIC         location = f'{self.configuration.MODEL_HISTORY_PATH}'
# MAGIC         class NpEncoder(json.JSONEncoder):
# MAGIC             def default(self, obj):
# MAGIC                 if isinstance(obj, np.integer):
# MAGIC                     return int(obj)
# MAGIC                 if isinstance(obj, np.floating):
# MAGIC                     return float(obj)
# MAGIC                 if isinstance(obj, np.ndarray):
# MAGIC                     return obj.tolist()
# MAGIC                 return super(NpEncoder, self).default(obj)
# MAGIC         json_string = json.dumps(model_summary, cls=NpEncoder)
# MAGIC         df = spark.read.json(sc.parallelize([json_string]))
# MAGIC         df = df.withColumn("time_stamp", current_timestamp())
# MAGIC 
# MAGIC         if not overwrite:
# MAGIC             if Utility.file_exists(dbutils, location):
# MAGIC                 df_existing = spark.read.parquet(location)
# MAGIC                 df = df.union(df_existing)
# MAGIC         df.coalesce(1).write.mode('overwrite').format('parquet').save(location)
# MAGIC 
# MAGIC     def get_params_summary(self, cvModel, additional_params):
# MAGIC         bestModel = cvModel.bestModel
# MAGIC         params = {}
# MAGIC         if isinstance(bestModel, LogisticRegression):
# MAGIC             params = {
# MAGIC                 'L1 regularization': bestModel.getRegParam(),
# MAGIC                 'L2 regularization': bestModel.getElasticNetParam(),
# MAGIC                 'iterations': bestModel.getMaxIter()
# MAGIC             }
# MAGIC         elif isinstance(bestModel, RandomForestClassifier):
# MAGIC             params = {
# MAGIC                 'Max Depth': bestModel.getMaxDepth(),
# MAGIC                 'No. of Trees': bestModel.getNumTrees()
# MAGIC             }
# MAGIC         else:
# MAGIC             params = {}
# MAGIC             
# MAGIC         if additional_params is not None:
# MAGIC             params.update(additional_params)
# MAGIC         return params
# MAGIC 
# MAGIC     def get_summary(self, model, model_family, model_name, \
# MAGIC                     dataset, training_time_seconds, class_imbalance_strategy, \
# MAGIC                     cross_validation_strategy, training, testing,
# MAGIC                     additional_params=None):
# MAGIC 
# MAGIC         test_prediction = model.transform(testing)
# MAGIC         training_prediction = model.transform(training)
# MAGIC 
# MAGIC         plots = {
# MAGIC             'training_loss_hires': f'{self.configuration.MODEL_PLOTS_BASE_PATH}/{model_name}_loss_hires.png',
# MAGIC             'training_loss_lowres': f'{self.configuration.MODEL_PLOTS_BASE_PATH}/{model_name}_loss_lowres.png',
# MAGIC             'test_roc_hires': f'{self.configuration.MODEL_PLOTS_BASE_PATH}/{model_name}_roc_hires.png',
# MAGIC             'test_roc_lowres': f'{self.configuration.MODEL_PLOTS_BASE_PATH}/{model_name}_roc_lowres.png', 
# MAGIC             'test_fmeasure_hires': f'{self.configuration.MODEL_PLOTS_BASE_PATH}/{model_name}_fMeasure_hires.png',
# MAGIC             'test_fmeasure_hires': f'{self.configuration.MODEL_PLOTS_BASE_PATH}/{model_name}_fMeasure_lowres.png',
# MAGIC             'coefficients_hires': f'{self.configuration.MODEL_PLOTS_BASE_PATH}/{model_name}_coefficients.png'
# MAGIC         }
# MAGIC         model_summary = {
# MAGIC             'model_family': model_family,
# MAGIC             'model': model_name,
# MAGIC             'training_dataset': dataset.training_dataset_name,
# MAGIC             'location': f'{self.configuration.TRAINED_MODEL_BASE_PATH}/{model_name}',
# MAGIC             'features': -1, #Not all models have this. Need to figure a better way to get this. model.bestModel.numFeatures,
# MAGIC             'parameters': self.get_params_summary(model, additional_params),
# MAGIC             'class_imbalance_strategy': class_imbalance_strategy,
# MAGIC             'cross_validation_strategy': cross_validation_strategy,
# MAGIC             'time_to_train': f'{training_time_seconds} Seconds',
# MAGIC             'training_prediction_summary': self.get_predictions_summary(training_prediction),
# MAGIC             'test_dataset': dataset.test_dataset_name,
# MAGIC             'test_prediction_summary': self.get_predictions_summary(test_prediction),
# MAGIC             'training_scores': self.get_training_scores(model),
# MAGIC             'test_scores': self.get_scores(test_prediction),
# MAGIC             'plots': plots
# MAGIC         }
# MAGIC         return model_summary
# MAGIC 
# MAGIC         

# COMMAND ----------

ls /dbfs/user/ram.senth@berkeley.edu/tmp

# COMMAND ----------

dbutils.fs.mv(f'dbfs:/user/ram.senth@berkeley.edu/tmp/{filename}', f'{configuration.MOUNTED_BLOB_STORE}/library/{filename}')

# COMMAND ----------

# MAGIC %%sh
# MAGIC cat /dbfs/mnt/team11-blobstore/library/training_summarizer_v01.py

# COMMAND ----------


