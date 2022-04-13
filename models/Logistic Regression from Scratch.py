# Databricks notebook source
# general imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from configuration_v01 import Configuration
from training_summarizer_v01 import TrainingSummarizer
from down_sampler_v02 import DownSampler
from custom_transformer_v05 import CustomTransformer

# magic commands
%matplotlib inline
%reload_ext autoreload
%autoreload 2

configuration = Configuration()



# COMMAND ----------

# To keep it simple, we will use a subset of features.
cols = ['DEP_DEL15', 'origin_weather_Present_Weather_Hail', 'origin_weather_Present_Weather_Storm',
        'dest_weather_Present_Weather_Hail', 'dest_weather_Present_Weather_Storm', 'prior_delayed', 'Holiday_5Day']

def load_data():
    """
        Load and prep both training and test datasets.
    """
    # Create Transformers
    # Custom sampler that down samples the majority class to address class imbalance.
    down_sampler = DownSampler(labelCol=configuration.orig_label_col, ratio=3.0)
    # Custom transformer that takes a smaller subset of samples to make training faster and also vectorizes the features.
    custom_transformer = CustomTransformer(percentage=0.0001, inputCols=cols)

    # Load training data.
    train = spark.read.parquet(configuration.TRANSFORMED_TRAINING_DATA)
    display(train.groupBy(configuration.orig_label_col).count())
    displayHTML("""Table 1: Original Training Dataset</p>""")

    # Training data transformations: Downsample majority class to address imbalance, take a subset and vectorize features.
    train_vectorized = custom_transformer.transform(down_sampler.transform(train))
    train_vectorized.cache()
    display(train_vectorized.groupBy(configuration.orig_label_col).count())
    displayHTML("""Table 2: Transformed Training Dataset""")

    # Load training data.
    test = spark.read.parquet(configuration.TRANSFORMED_2019_DATA)
    display(test.groupBy(configuration.orig_label_col).count())
    displayHTML("""Table 3: Original Test Dataset</p>""")

    # Test data transformations: Transformations: take a subset and vectorize features.
    test_vectorized = custom_transformer.transform(test)
    display(test_vectorized.groupBy(configuration.orig_label_col).count())
    displayHTML("""Table 4: Transformed Test Dataset</p>""")
    test_vectorized.cache()
    return train_vectorized, test_vectorized


# COMMAND ----------

def get_confusion_matrix_vals(y_vals):
    """
        Calculate true positive, true negatives, false positives, and false negatives for a logistic regression model
        Input: (y_pred, y_true)
    """
    output = ['tp', 'tn', 'fp', 'fn']
    y_pred = y_vals[0]  # int(y_vals[0] > 0.5)
    y_true = y_vals[1]
    yield ('tp', y_true*y_pred)
    yield ('tn', (1-y_true)*(1-y_pred))
    yield ('fp', (1-y_true)*y_pred)
    yield ('fn', y_true*(1-y_pred))
    

def f_beta_loss(dataRDD, W):
    """
    Compute differentiable f_beta_loss.
    Args:
        dataRDD - each record is a tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    """
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1]))
    epsilon = 1e-5
    beta = 0.5
    
    # F_beta loss
    cm = augmentedData.map(lambda x: (1 / (1 + np.exp(-x[0].dot(W))), x[1])).flatMap(get_confusion_matrix_vals).reduceByKey(lambda x, y: x+y).cache()
    cm = dict(cm.collect())
    
    precision = cm['tp'] / (cm['tp'] + cm['fp'] + epsilon)
    recall = cm['tp'] / (cm['tp'] + cm['fn'] + epsilon)
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + epsilon)
    f_beta_loss = 1-f_beta

    # F_beta loss as Dice loss
#     loss = augmentedData.map(lambda x: (1 / (1 + np.exp(-x[0].dot(W))), x[1])).map(lambda x: 1 - ((1 + beta**2) * x[1] * x[0])/(beta**2 * x[1] + x[0])).mean()
    
    # accuracy
    loss = augmentedData.map(lambda x: (-x[1]*(np.log(1 / (1 + np.exp(-x[0].dot(W))) + epsilon))\
                            -((1-x[1])*(np.log(1 - (1 / (1 + np.exp(-x[0].dot(W)))) + epsilon))))).mean()

    return loss, f_beta_loss

def GDUpdate(dataRDD, W, learningRate = 0.1):
    """
    Perform one OLS gradient descent step/update.
    Args:
        dataRDD - records are tuples of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    Returns:
        new_model - (array) updated coefficients, bias at index 0
    """
    # add a bias 'feature' of 1 at index 0
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1])).cache()
    
    beta = 0.5
    
    # gradient for accuracy
    grad = augmentedData.map(lambda x: (1 / (1 + np.exp(-x[0].dot(W))) - x[1])*x[0]).mean()
    
    # gradient for Dice loss
#     grad = augmentedData.map(lambda x: ((-(1 + beta**2) * x[1]**2 * (1 / (1 + np.exp(-x[0].dot(W))))*(1 - (1 / (1 + np.exp(-x[0].dot(W)))))) /  ((beta**2)*(1 / (1 + np.exp(-x[0].dot(W)))) + x[1])**2)*x[0]).mean() 

    new_model = W - learningRate*grad

   
    return new_model

def GradientDescent(trainRDD, testRDD, wInit, nSteps = 20, learningRate = 0.1, verbose = False):
    """
    Perform nSteps iterations of OLS gradient descent and 
    track loss on a test and train set. Return lists of
    test/train loss and the models themselves.
    """
    # initialize lists to track model performance
    train_history, test_history, train_f_beta_history, test_f_beta_history, model_history = [], [], [], [], []
    
    # perform n updates & compute test and train loss after each
    model = wInit
    for idx in range(nSteps): 
        bModel = sc.broadcast(model)
        model = GDUpdate(trainRDD, bModel.value, learningRate)
        training_loss, training_f_beta_loss = f_beta_loss(trainRDD, model) 
        test_loss, test_f_beta_loss = f_beta_loss(testRDD, model)
        
        # keep track of test/train loss for plotting
        train_history.append(training_loss)
        train_f_beta_history.append(training_f_beta_loss)
        test_history.append(test_loss)
        test_f_beta_history.append(test_f_beta_loss)
        model_history.append(model)
        
        # console output if desired
        if verbose:
            print("----------")
            print(f"STEP: {idx+1}")
            print(f"training loss: {training_loss}")
            print(f"test loss: {test_loss}")
            print(f"training f_beta: {training_f_beta_loss}")
            print(f"test f_beta: {test_f_beta_loss}")
            print(f"Model: {[round(w,3) for w in model]}")
    return train_history, test_history, train_f_beta_history, test_f_beta_history, model_history

def plotErrorCurves(trainLoss, testLoss, trainFBeta, testFBeta, title = None):
    """
    Helper function for plotting.
    Args: trainLoss (list of errors) , testLoss (list of errors)
    """
    fig, ax = plt.subplots(1,1)
    x = list(range(len(trainLoss)))[1:]
    ax.plot(x, trainLoss[1:], 'k--', label='Training Loss (Accuracy)')
    ax.plot(x, testLoss[1:], 'r--', label='Test Loss (Accuracy)')
    ax.plot(x, trainFBeta[1:], 'b--', label='Training Loss (F-Beta)')
    ax.plot(x, testFBeta[1:], 'c--', label='Test Loss (F-Beta)')
    ax.legend(loc='right', fontsize='x-large')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    fig.savefig(f'{configuration.MODEL_PLOTS_BASE_PATH}/lr_manual_accuracy_f_beta_low')
    if title:
        plt.title(title)
    plt.show()

# COMMAND ----------

def run_regression():
    train_vectorized, test_vectorized = load_data()
    
    # Initialize weights/baseline model.
    BASELINE = np.array([0.2] + [0]*(len(cols) - 1))

    train_history, test_history, train_FBeta, test_FBeta, model_history = GradientDescent(train_vectorized.rdd, \
                                                                                      test_vectorized.rdd, \
                                                                                      BASELINE, nSteps=15, \
                                                                                      learningRate=0.25, verbose=True)
    plotErrorCurves(train_history, test_history, train_FBeta, test_FBeta, title = 'Logistic Regression Gradient Descent')

run_regression()

# COMMAND ----------

display(spark.read.parquet(configuration.MODEL_HISTORY_PATH))

# COMMAND ----------

# MAGIC %md # Display Parameters and Metrics For All Models

# COMMAND ----------

from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel, RandomForestClassifier, RandomForestClassificationModel
from pyspark.sql.functions import *
from pyspark.sql.window import Window

def get_params(model, location):
    cvModel = CrossValidatorModel.load(location)
    bestModel = cvModel.bestModel
    if isinstance(bestModel, LogisticRegressionModel):
        params = {
            'L1 regularization': str(bestModel.getRegParam()),
            'L2 regularization': str(bestModel.getElasticNetParam()),
            'iterations': str(bestModel.getMaxIter()),
            'num_features': str(bestModel.numFeatures)
        }
    elif isinstance(bestModel, RandomForestClassificationModel):
        params = {
            'Max Depth': str(bestModel.getMaxDepth()),
            'No. of Trees': str(bestModel.getNumTrees),
            'num_features': str(bestModel.numFeatures)
        }
    else:
        params = {}
        
    return params

def get_params_for_all_models():
    windowSpec  = Window.partitionBy("model").orderBy(col("time_stamp").desc())
    df_model_summary = spark.read.parquet(configuration.MODEL_HISTORY_PATH).filter(col('model') != 'lr_toy')
    df_latest = df_model_summary.withColumn("row_number",row_number().over(windowSpec)).filter(col('row_number') == 1)
    
    # iterate through the models, get parameters and display.
    model_locs = df_latest.select('model', 'location', 'class_imbalance_strategy', \
                                  'cross_validation_strategy', 'plots', 'test_prediction_summary', \
                                  'training_scores', 'test_scores', 'time_to_train', 'time_stamp').toPandas()
    params = []
    for index, row in model_locs.iterrows():
        params.append(get_params(row["model"], row["location"]))
    model_locs['params'] = params
    display(model_locs)
        
get_params_for_all_models()

# COMMAND ----------


