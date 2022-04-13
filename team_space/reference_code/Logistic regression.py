# Databricks notebook source
# MAGIC %md
# MAGIC # Logistic Regression
# MAGIC Compare sklearn, spark ML, and homegrown GD

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### sklearn

# COMMAND ----------

# ML modules
#importing all the required ML packages
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as LogisticRegression_skl
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix


# COMMAND ----------

# MAGIC %md
# MAGIC ### Spark

# COMMAND ----------

blob_container = "w261team11" # The name of your container created in https://portal.azure.com
storage_account = "w261sa" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261team11" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261team11key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)


# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset:
# MAGIC ### Titanic - just going to clean it up for training. Not doing anything special.
# MAGIC https://www.kaggle.com/c/titanic/data

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install fsspec

# COMMAND ----------


data=pd.read_csv('/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/titanic/train.csv')

# COMMAND ----------

data.head()

# COMMAND ----------

data['Age_band']=0
data.loc[data['Age']<=16,'Age_band']=0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[data['Age']>64,'Age_band']=4
data.head(2)

# COMMAND ----------

data['Fare_cat']=0
data.loc[data['Fare']<=7.91,'Fare_cat']=0
data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1
data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2
data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3
data.head(2)

# COMMAND ----------

data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data.head(2)

# COMMAND ----------

data.columns

# COMMAND ----------

data.drop(['Name','Age','Ticket','Fare','Cabin','PassengerId'],axis=1,inplace=True)

# COMMAND ----------

data['Embarked'].fillna(0,inplace=True)

# COMMAND ----------

data.head(2)

# COMMAND ----------

train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['Survived']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalize

# COMMAND ----------

cols = train_X.columns
# "Pclass", "Sex", "SibSp","Parch","Embarked","Age_band","Fare_cat"
len(train_X)

# COMMAND ----------

scaler = preprocessing.StandardScaler().fit(train[cols])
train[cols] = scaler.transform(train[cols]) 
test[cols] = scaler.transform(test[cols]) 

# COMMAND ----------

train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]

# COMMAND ----------

train_X.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC # TRAIN MODELS

# COMMAND ----------

# MAGIC %md
# MAGIC ## SKLearn l-BFGS

# COMMAND ----------

# PARAMS:
# penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’, verbose=0, warm_start=False, n_jobs=None

model = LogisticRegression_skl(max_iter=100, C=10.0, penalty='l2', solver="lbfgs")
model.fit(train_X,train_Y)
sk_pred = model.predict(test_X)


# COMMAND ----------

# testing accuracy
metrics.accuracy_score(sk_pred,test_Y)

# COMMAND ----------

# Print the coefficients and intercept for logistic regression, and save for later
SKL_lBFGS_Coefficients = model.coef_[0]
SKL_lBFGS_Intercept = model.intercept_[0]
print("Coefficients: " + str(SKL_lBFGS_Coefficients))
print("Intercept: " + str(SKL_lBFGS_Intercept))

# COMMAND ----------

# MAGIC %md
# MAGIC ## SKLearn SGD

# COMMAND ----------

clf = SGDClassifier(max_iter=100, loss='log', penalty='l2', alpha=0.1)
clf.fit(train_X,train_Y)

# COMMAND ----------

SGD_Coefficients = clf.coef_[0]
SGD_Intercept = clf.intercept_[0]
print("Coefficients: " + str(SGD_Coefficients))
print("Intercept: " + str(SGD_Intercept))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark ML

# COMMAND ----------

# Load training data
training = spark.createDataFrame(train)
testing = spark.createDataFrame(test)

# COMMAND ----------

# vectorize features
training = VectorAssembler(inputCols=["Pclass", "Sex", "SibSp","Parch","Embarked","Age_band","Fare_cat"], outputCol="features").transform(training)
testing = VectorAssembler(inputCols=["Pclass", "Sex", "SibSp","Parch","Embarked","Age_band","Fare_cat"], outputCol="features").transform(testing)

# COMMAND ----------

lr = LogisticRegression(featuresCol="features", labelCol="Survived", maxIter=100, regParam=0.1, elasticNetParam=0)
# elasticNetParam: For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.

# Fit the model
lrModel = lr.fit(training)

# COMMAND ----------

# Print the coefficients and intercept for logistic regression, and save for later
SML_Coefficients = ast.literal_eval(str(lrModel.coefficients))
SML_Intercept = lrModel.intercept
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# COMMAND ----------

# traing accuracy
lrModel.summary.accuracy

# COMMAND ----------

result = lrModel.transform(testing)

# COMMAND ----------

# testing accuracy
lrModel.evaluate(testing).accuracy

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark MLLib

# COMMAND ----------

# lr_sgd = LogisticRegressionWithSGD(featuresCol="features", labelCol="Survived", maxIter=100, regParam=0.1, elasticNetParam=0)
# elasticNetParam: For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
trainingRDD = spark.createDataFrame(train).rdd.map(lambda row: LabeledPoint(row[0], row[1:])).cache()
testingRDD = spark.createDataFrame(test).rdd.map(lambda row: LabeledPoint(row[0], row[1:])).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### LogisticRegressionWithLBFGS

# COMMAND ----------

SMLLib_LBFGS = LogisticRegressionWithLBFGS.train(trainingRDD, regType='l2', regParam=0.1, iterations=100, intercept=True)

# COMMAND ----------

SMLLib_LBFGS_Coefficients = SMLLib_LBFGS.weights
SMLLib_LBFGS_Intercept = SMLLib_LBFGS.intercept
print("Coefficients: " + str(SMLLib_LBFGS.weights))
print("Intercept: " + str(SMLLib_LBFGS.intercept))

# COMMAND ----------

#Save model, reload model and score against the test dataset.
def test_save_load(trained_model, test_data):
#     import LogisticRegressionModel from pyspark.mllib.classification
    from pyspark.mllib import classification 
    # Save model
    model_loc = '/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/titanic/mllib_lbfgs'
    model_loc = f"{blob_url}/model/trained/titanic"
    trained_model.save(spark, model_loc)
    reloaded_model = classification.LogisticRegressionModel.load(spark, model_loc)
    # evaluate against test data
    print(reloaded_model.predict(test_data))

test_save_load(SMLLib_LBFGS, testingRDD)

# COMMAND ----------

# MAGIC %md
# MAGIC ### LogisticRegressionWithSGD

# COMMAND ----------

SMLLib_SGD = LogisticRegressionWithSGD.train(trainingRDD, regType='l2', regParam=0.1, iterations=100, intercept=True)

# COMMAND ----------

SMLLib_SGD_Coefficients = SMLLib_SGD.weights
SMLLib_SGD_Intercept = SMLLib_SGD.intercept
print("Coefficients: " + str(SMLLib_SGD.weights))
print("Intercept: " + str(SMLLib_SGD.intercept))

# COMMAND ----------

# MAGIC %md
# MAGIC __Evaluation (AOC):__
# MAGIC https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#module-pyspark.mllib.evaluation

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark RDD - homegrown

# COMMAND ----------

import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# COMMAND ----------

def LogLoss(dataRDD, W): 
    """
    Compute logistic loss error.
    Args:
        dataRDD - each record is a tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    """
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1]))
    ################## YOUR CODE HERE ##################
    loss = augmentedData.map(lambda x: x[1]*np.log(sigmoid(W.dot(x[0]))) + (1-x[1])*np.log(1 - sigmoid(W.dot(x[0])))).mean()*-1
   
    ################## (END) YOUR CODE ##################
    return loss

# COMMAND ----------


def GDUpdate_wReg(dataRDD, W, learningRate = 0.1, regType='ridge', regParam = 0.1):
    """
    Perform one gradient descent step/update with ridge or lasso regularization.
    Args:
        dataRDD - tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
        learningRate - (float) defaults to 0.1
        regType - (str) 'ridge' or 'lasso', defaults to None
        regParam - (float) regularization term coefficient
    Returns:
        model   - (array) updated coefficients, bias still at index 0
    """
    # augmented data
    N=dataRDD.count()
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1]))
    
    new_model = None
    #################### YOUR CODE HERE ###################
    # Use the same way as before to find the first component of the gradient function
    # Note: the official equation should be a sum() which provides a lower log loss and a more accurate prediction.
    # However, Spark ML uses mean() for their calculation, possibly to avoid overflow issues.
    
    grad = augmentedData.map(lambda x: ((sigmoid(W.dot(x[0])) - x[1])*x[0])).sum()
    if regType == 'ridge':
        grad += regParam * np.append([0.0], W[1:])
    elif regType == 'lasso':
        grad += regParam * np.append([0.0], np.sign(W)[1:])
    new_model = W - learningRate * grad/N
    ################## (END) YOUR CODE ####################
    return new_model

# COMMAND ----------

# part d - ridge/lasso gradient descent function
def GradientDescent_wReg(trainRDD, testRDD, wInit, nSteps = 100, learningRate = 0.1,
                         regType='ridge', regParam = 0.1, verbose = False):
    """
    Perform nSteps iterations of regularized gradient descent and 
    track loss on a test and train set. Return lists of
    test/train loss and the models themselves.
    """
    # initialize lists to track model performance
    train_history, test_history, model_history = [], [], []
    
    # perform n updates & compute test and train loss after each
    model = wInit
    for idx in range(nSteps):  
        # update the model
        model = GDUpdate_wReg(trainRDD, model, learningRate, regType, regParam)
        
        # keep track of test/train loss for plotting
        train_history.append(LogLoss(trainRDD, model))
        test_history.append(LogLoss(testRDD, model))
        model_history.append(model)
        
        # console output if desired
        if verbose:
            print("----------")
            print(f"STEP: {idx+1}")
            print(f"Model: {[round(w,3) for w in model]}")
    return train_history, test_history, model_history

# COMMAND ----------

trainRDD = spark.createDataFrame(train).rdd.map(lambda x: (x[1:],x[0]))
testRDD = spark.createDataFrame(test).rdd.map(lambda x: (x[1:],x[0]))

# COMMAND ----------

wInit = np.random.uniform(0,1,8)
ridge_results = GradientDescent_wReg(trainRDD, testRDD, wInit, nSteps = 500, regType='ridge', regParam = 0.1 )

# COMMAND ----------

# intercept and coefficients
HG_Coefficients = ridge_results[2][-1][1:]
HG_Intercept = ridge_results[2][-1][:1][0]
print("Intercept: " + str(HG_Intercept))
print("Coefficients: " + str(HG_Coefficients))

# COMMAND ----------

def plotErrorCurves(trainLoss, testLoss, title = None):
    """
    Helper function for plotting.
    Args: trainLoss (list of MSE) , testLoss (list of MSE)
    """
    fig, ax = plt.subplots(1,1,figsize = (16,8))
    x = list(range(len(trainLoss)))[1:]
    ax.plot(x, trainLoss[1:], 'k--', label='Training Loss')
    ax.plot(x, testLoss[1:], 'r--', label='Test Loss')
    ax.legend(loc='upper right', fontsize='x-large')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    if title:
        plt.title(title)
    plt.show()

# COMMAND ----------

plotErrorCurves(ridge_results[0],ridge_results[1])

# COMMAND ----------

# predict probabilities for homegrown
w = ridge_results[2][-1] # final model
augmentedTestData = testRDD.map(lambda x: (np.append([1.0], x[0]), x[1]))
results = augmentedTestData.map(lambda x: (sigmoid(w.dot(x[0])),x[1])).collect()

# COMMAND ----------

# set prediction to 1 if probability is greater than or equal to 0.5, and 0 otherwise.
df = pd.DataFrame(results)
df['pred'] = df[0] >= .5

# COMMAND ----------

# testing accuracy
sum(df['pred']==df[1])/len(df)

# COMMAND ----------

# testing accuracy
sum(df['pred']==df[1])/len(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MODELS

# COMMAND ----------

# Pretty Print
print("{:15s}\t| {:15s}\t| {:15s}\t| {:15s}\t| {:15s}\t| {}" \
      .format("Homegrown",
              "Spark ML (l-BFGS)",
              "MLLib l-BFGS",
              "MLLib SGD",
              "SKLearn SGD",
              "SKLearn l-BFGS"
             ))
print("="*130)
print("Coefficients")   
print("-"*130)  
for r in range(7):
    print ("{:15.8f}\t| {:15.8f}\t| {:15.8f}\t| {:15.8f}\t| {:15.8f}\t| {:15.8f}" \
           .format(HG_Coefficients[r],
                   SML_Coefficients[r],
                   SMLLib_LBFGS_Coefficients[r],
                   SMLLib_SGD_Coefficients[r],
                   SGD_Coefficients[r],
                   SKL_lBFGS_Coefficients[r]
                  ))
print("-"*130)    
print("Intercept")   
print("-"*130)   
print ("{:15.8f}\t| {:15.8f}\t| {:15.8f}\t| {:15.8f}\t| {:15.8f}\t| {:15.8f}" \
       .format(HG_Intercept,
               SML_Intercept,
               SMLLib_LBFGS_Intercept,
               SMLLib_SGD_Intercept,
               SGD_Intercept,
               SKL_lBFGS_Intercept
              ))    


# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC Homegrown coefficients resemble Spark ML when the update rule is:
# MAGIC $$
# MAGIC w_{t+1} = w_t - \alpha\bigg[\frac{1}{m}\bigg(\sum^{m}_{i=1}(sigmoid(w^Tx_i) - y_i)*x_i\bigg) + \beta w\bigg]
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC Homegrown coefficients resemble SKLearn l-BFGS when the update rule is (probably by accident, as it doesn't make sense to divide the reg term by m):
# MAGIC $$
# MAGIC w_{t+1} = w_t - \alpha\bigg[\frac{1}{m}\bigg(\sum^{m}_{i=1}(sigmoid(w^Tx_i) - y_i)*x_i + \beta w \bigg)\bigg]
# MAGIC $$
# MAGIC 
# MAGIC This is the way Andrew Ng illustrates it in his lecture [3.4.3 Regularized Linear Regression](https://www.youtube.com/watch?v=GhXojUkyIkQ). Most literature does not divide the reg term by N, for example, these [Stanford lecture notes](http://theory.stanford.edu/~tim/s16/l/l6.pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gradient Descent VS Coordinate Descent
# MAGIC Coordinate descent updates one parameter at a time, while gradient descent attempts to update all parameters at once. It's hard to specify exactly when one algorithm will do better than the other.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Coordinate descent

# COMMAND ----------

# MAGIC %md
# MAGIC - Coordinate descent slides http://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/coordinate_descent.pdf
# MAGIC - unified view, Wright 2015:  https://arxiv.org/pdf/1502.04759.pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ### SKLEARN 
# MAGIC ### Logistic regression
# MAGIC https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

# COMMAND ----------

# MAGIC %md
# MAGIC Logistic regression, despite its name, is a linear model for classification rather than regression. Logistic regression is also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.
# MAGIC 
# MAGIC The implementation of logistic regression in scikit-learn can be accessed from class LogisticRegression. This implementation can fit binary, One-vs- Rest, or multinomial logistic regression with optional L2 or L1 regularization.

# COMMAND ----------

# MAGIC %md
# MAGIC - LIBLINEAR – A Library for Large Linear Classification https://www.csie.ntu.edu.tw/~cjlin/liblinear/
# MAGIC - SAG – Mark Schmidt, Nicolas Le Roux, and Francis Bach Minimizing Finite Sums with the Stochastic Average Gradient https://hal.inria.fr/hal-00860051/document
# MAGIC - SAGA – Defazio, A., Bach F. & Lacoste-Julien S. (2014). SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives https://arxiv.org/abs/1407.0202
# MAGIC - Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent methods for logistic regression and maximum entropy models. Machine Learning 85(1-2):41-75. https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf

# COMMAND ----------

# MAGIC %md
# MAGIC The solver “liblinear” uses a __coordinate descent (CD)__ algorithm, and relies on the excellent C++ LIBLINEAR library, which is shipped with scikit-learn. However, the CD algorithm implemented in liblinear cannot learn a true multinomial (multiclass) model; instead, the optimization problem is decomposed in a “one-vs-rest” fashion so separate binary classifiers are trained for all classes. This happens under the hood, so LogisticRegression instances using this solver behave as multiclass classifiers. For L1 penalization sklearn.svm.l1_min_c allows to calculate the lower bound for C in order to get a non “null” (all feature weights to zero) model.
# MAGIC 
# MAGIC The “lbfgs”, “sag” and “newton-cg” solvers only support L2 penalization and are found to converge faster for some high dimensional data. Setting multi_class to “multinomial” with these solvers learns a true multinomial logistic regression model [5], which means that its probability estimates should be better calibrated than the default “one-vs-rest” setting.
# MAGIC 
# MAGIC The “sag” solver uses a Stochastic Average Gradient descent [6]. It is faster than other solvers for large datasets, when both the number of samples and the number of features are large.
# MAGIC 
# MAGIC The “saga” solver [7] is a variant of “sag” that also supports the non-smooth penalty=”l1” option. This is therefore the solver of choice for sparse multinomial logistic regression.
# MAGIC 
# MAGIC The “lbfgs” is an optimization algorithm that approximates the Broyden–Fletcher–Goldfarb–Shanno algorithm [8], which belongs to quasi-Newton methods (second order method - see below). The “lbfgs” solver is recommended for use for small data-sets but for larger datasets its performance suffers. [9]
# MAGIC 
# MAGIC __References:__
# MAGIC 
# MAGIC - [5]	Christopher M. Bishop: Pattern Recognition and Machine Learning, Chapter 4.3.4
# MAGIC - [6]	Mark Schmidt, Nicolas Le Roux, and Francis Bach: Minimizing Finite Sums with the Stochastic Average Gradient. https://hal.inria.fr/hal-00860051/document
# MAGIC - [7]	Aaron Defazio, Francis Bach, Simon Lacoste-Julien: SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives. https://arxiv.org/abs/1407.0202
# MAGIC - [8]	https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
# MAGIC - [9]	“Performance Evaluation of Lbfgs vs other solvers” http://www.fuzihao.org/blog/2016/01/16/Comparison-of-Gradient-Descent-Stochastic-Gradient-Descent-and-L-BFGS/

# COMMAND ----------

# MAGIC %md
# MAGIC ### SKLEARN 
# MAGIC ### Stochastic Gradient Descent - SGD
# MAGIC https://scikit-learn.org/stable/modules/linear_model.html#stochastic-gradient-descent-sgd
# MAGIC 
# MAGIC Stochastic gradient descent is a simple yet very efficient approach to fit linear models. It is particularly useful when the number of samples (and the number of features) is very large. The partial_fit method allows online/out-of-core learning.
# MAGIC 
# MAGIC The classes SGDClassifier and SGDRegressor provide functionality to fit linear models for classification and regression using different (convex) loss functions and different penalties. E.g., with loss="log", SGDClassifier fits a logistic regression model, while with loss="hinge" it fits a linear support vector machine (SVM).

# COMMAND ----------

# MAGIC %md
# MAGIC ### l-BFGS

# COMMAND ----------

# MAGIC %md
# MAGIC https://www.coursera.org/lecture/machine-learning-applications-big-data/how-to-train-algorithms-second-order-methods-IMB0F

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="second-order-methods.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC paraboloid vs hyperplane that is tangent to the last solution

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC TODO: Look at team silibon for nice expo on lbfgs and gd (SP19-5-Siilbon)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

import numpy as np
theta=np.array([1,0])
x=np.array([1,1])
y=2
grad = (2)*(np.dot(theta,x.T)-y)*x.T
print(grad)
ntheta = theta.T - 0.1*(grad)
print(ntheta)

# COMMAND ----------



# COMMAND ----------


