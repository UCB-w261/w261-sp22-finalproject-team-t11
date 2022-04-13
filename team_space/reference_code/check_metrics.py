# Databricks notebook source
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, LogisticRegression
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql.functions import col, isnan, when, count, lit, explode, array

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

train = data_preparer.prep_data(spark.read.parquet(configuration.TRANSFORMED_TRAINING_DATA))
test_2019 = data_preparer.prep_data(spark.read.parquet(configuration.TRANSFORMED_2019_DATA))
test_2020 = data_preparer.prep_data(spark.read.parquet(configuration.TRANSFORMED_2020_DATA))
test_2021 = data_preparer.prep_data(spark.read.parquet(configuration.TRANSFORMED_2021_DATA))


# COMMAND ----------

training_withsplits, training_prepd, testing_prepd = data_preparer.load_and_prep_data(spark, configuration.FULL_DATASET,\
                                                                          'None',
                                                                          apply_class_weights=False)

# COMMAND ----------

cv_model_lr = load_model(TRAINED_LR_MODEL_GRIDSEARCH, CrossValidatorModel)
model_lr = cv_model_lr.bestModel

# COMMAND ----------

# train_pred_lr = model_lr.transform(train)
train_pred_lr = cv_model_lr.transform(train)

# COMMAND ----------

display(spark.read.parquet(configuration.MODEL_HISTORY_PATH).filter(col('model') == 'lr_gridsearch'))

# COMMAND ----------

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

def from_err_analysis_code(predictions):
    tp = predictions.select('label','prediction').where((predictions['label']==1) & (predictions['prediction']==1)).count()
    tn = predictions.select('label','prediction').where((predictions['label']==0) & (predictions['prediction']==0)).count()
    fp = predictions.select('label','prediction').where((predictions['label']==0) & (predictions['prediction']==1)).count()
    fn = predictions.select('label','prediction').where((predictions['label']==1) & (predictions['prediction']==0)).count()

    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn)
    f1_score = get_f1_score(tp, fp, fn)
    f_beta = get_f_beta(precision, recall)
    accuracy = get_accuracy(tp, tn, fp, fn)

    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1_score,
        'F_beta': f_beta,
        'Accuracy': accuracy
    }
    
def from_library_code(predictions):
    grouped = predictions.groupby(configuration.label_col, 'prediction').count().toPandas()
    return {
        'TP': grouped[(grouped['label'] == 1) & (grouped['prediction'] == 1)].iat[0, 2],
        'TN': grouped[(grouped['label'] == 0) & (grouped['prediction'] == 0)].iat[0, 2],
        'FP': grouped[(grouped['label'] == 0) & (grouped['prediction'] == 1)].iat[0, 2],
        'FN': grouped[(grouped['label'] == 1) & (grouped['prediction'] == 0)].iat[0, 2]
    }

from sklearn.metrics import classification_report, accuracy_score, fbeta_score

def get_sklearn_scores(predictions):
    preds = predictions.select([configuration.label_col, 'prediction'])
#         df = pd.DataFrame(preds, columns =['probability', self.configuration.label_col, 'prediction'])
    df = preds.toPandas()
    y_true = df[configuration.label_col]
    y_pred = df['prediction']
    final_report = classification_report(y_true, y_pred, output_dict=True)
    f_beta = fbeta_score(y_true, y_pred, average=None, beta=0.5)
    weighted_accuracy = accuracy_score(y_true, y_pred)    

    return {
        'Accuracy': weighted_accuracy,
        'Precision': final_report.get('weighted avg').get('precision'),
        'Recall': final_report.get('weighted avg').get('recall'),
        'F_score (beta=1)': final_report.get('weighted avg').get('f1-score'),
        'F_beta (beta=0.5)': f_beta
    }
    

# COMMAND ----------

# MAGIC %md # Training Data

# COMMAND ----------

total = train_pred_lr.count()

tp = train_pred_lr.select('label','prediction').where((train_pred_lr['label']==1) & (train_pred_lr['prediction']==1)).count()
tn = train_pred_lr.select('label','prediction').where((train_pred_lr['label']==0) & (train_pred_lr['prediction']==0)).count()
fp = train_pred_lr.select('label','prediction').where((train_pred_lr['label']==0) & (train_pred_lr['prediction']==1)).count()
fn = train_pred_lr.select('label','prediction').where((train_pred_lr['label']==1) & (train_pred_lr['prediction']==0)).count()

# COMMAND ----------

# Original
# {"FN": 2463351, "FP": 472391, "TN": 5366473, "TP": 1496141}
# {"Accuracy": 0.6991012019038854, "F_beta__beta_0.5_": 0.685797342522396, "F_score__beta_1_": 0.6714135503530436, "weightedFalsePositiveRate": 0.39756234050445316, "weightedPrecision": 0.7157152661848951, "weightedRecall": 0.6991012019038855, "weightedTruePositiveRate": 0.6991012019038855}

#From this cell:
# "FN": 2463351, "FP": 1418094, "TN": 16091270, "TP": 1496141
print('Original: {"FN": 2463351, "FP": 472391, "TN": 5366473, "TP": 1496141}')
print(f'"FN": {fn}, "FP": {fp}, "TN": {tn}, "TP": {tp}')
# TP and FN are ok.
# FP and TN do not match.

# COMMAND ----------

print('original: {"TP": 1496141, "TN": 5366473, "FP": 472391, "FN": 2463351}')
print(from_library_code(train_pred_lr))

# COMMAND ----------


print(from_err_analysis_code(train_pred_lr))


# COMMAND ----------

# MAGIC %md # Using Prepd Data

# COMMAND ----------

train_pred_lr_prepd = cv_model_lr.transform(training_prepd)

# COMMAND ----------

print(from_library_code(train_pred_lr_prepd))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md #Using Test Data

# COMMAND ----------

# Original {"FN": 751243, "TN": 4762814, "FP": 432879, "TP": 470953}
test_pred_lr_2019 = cv_model_lr.transform(test_2019)

# COMMAND ----------

print(from_library_code(test_pred_lr_2019))

# COMMAND ----------

print(from_err_analysis_code(train_pred_lr_2019))

# COMMAND ----------

print(get_sklearn_scores(train_pred_lr_2019))

# COMMAND ----------



# COMMAND ----------


