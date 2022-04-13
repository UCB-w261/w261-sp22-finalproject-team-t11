# Databricks notebook source
# MAGIC %md # Helper Class For Preparing Data For Training
# MAGIC This class holds all the common methods to be used for preparing data for model training.

# COMMAND ----------

filename = 'data_preparer_v06.py'
from configuration_v01 import Configuration
configuration = Configuration()


# COMMAND ----------

# MAGIC %%writefile /dbfs/user/ram.senth@berkeley.edu/tmp/data_preparer_v06.py
# MAGIC from pyspark.ml.linalg import DenseVector, SparseVector, Vectors
# MAGIC from pyspark.ml.feature import VectorAssembler, StandardScaler
# MAGIC import pyspark.sql.functions as F
# MAGIC 
# MAGIC class DataPreparer():
# MAGIC     def __init__(self, configuration):
# MAGIC         self.configuration = configuration
# MAGIC 
# MAGIC     def version():
# MAGIC         return '6.0'
# MAGIC 
# MAGIC     def load_and_prep_data(self, spark, dataset, cv_strategy, apply_class_weights=False, checkpoint=True):
# MAGIC         # Load training data
# MAGIC         training = spark.read.parquet(dataset.training_dataset)
# MAGIC 
# MAGIC         # Apply data transformations needed for training.
# MAGIC         training = self.prep_data(training)
# MAGIC 
# MAGIC         if checkpoint:
# MAGIC             # Check point the data for optimization.
# MAGIC             training.checkpoint(eager=True)
# MAGIC 
# MAGIC         # Apply undersampling of majority class strategy for addressing class imbalance.
# MAGIC         balanced_training = self.address_imbalance(training, apply_class_weights)
# MAGIC         if checkpoint:
# MAGIC             # Check point the data for optimization.
# MAGIC             balanced_training.checkpoint(eager=True)
# MAGIC 
# MAGIC         # Apply cross validation strategy and prepare the training dataset with required folds.
# MAGIC         if cv_strategy == 'None':
# MAGIC             training_withsplits = CVSplitter.CV_noCV(balanced_training)
# MAGIC         elif cv_strategy == 'Rolling Window':
# MAGIC             training_withsplits = CVSplitter.CV_rolling_window(balanced_training)
# MAGIC         elif cv_strategy == 'Block Years':
# MAGIC             training_withsplits = CVSplitter.CV_block_by_year(balanced_training)
# MAGIC         else:
# MAGIC             training_withsplits = CVSplitter.CV_block_by_quarter(balanced_training)
# MAGIC 
# MAGIC         # Load test data
# MAGIC         testing = spark.read.parquet(dataset.test_dataset)
# MAGIC 
# MAGIC         # Apply data transformations needed for using the model.
# MAGIC         testing = self.prep_data(testing)
# MAGIC         
# MAGIC         if checkpoint:
# MAGIC             testing.checkpoint(eager=True)
# MAGIC             
# MAGIC         return training_withsplits, training, testing
# MAGIC 
# MAGIC     def prep_data(self, data):
# MAGIC         # vectorize features
# MAGIC         if self.configuration.features_col in data.columns:
# MAGIC             data = data.drop(self.configuration.features_col)
# MAGIC         vector_assembler = VectorAssembler(inputCols=self.configuration.feature_cols, outputCol=self.configuration.features_col)
# MAGIC         new_data = vector_assembler \
# MAGIC             .transform(data) \
# MAGIC             .withColumnRenamed(self.configuration.orig_label_col, self.configuration.label_col)
# MAGIC         return new_data
# MAGIC       
# MAGIC     def address_imbalance(self, df, apply_class_weights):
# MAGIC         def add_class_weights(df):
# MAGIC             negatives = df.filter(df[self.configuration.label_col]==0).count()
# MAGIC             positives = df.filter(df[self.configuration.label_col]==1).count()
# MAGIC             balance_ratio = negatives / (positives+negatives)
# MAGIC             return df.withColumn(self.configuration.class_weights_col, when(training[self.configuration.label_col] == 1, balance_ratio).otherwise(1-balance_ratio))
# MAGIC 
# MAGIC         def oversample_minority_class(df):
# MAGIC             minor_df = df.where(df[self.configuration.label_col] == 1)
# MAGIC             major_df = df.where(df[self.configuration.label_col] == 0)
# MAGIC             # ratio = int(negatives/positives) # defined in source
# MAGIC             ratio = 3 # for experimentation, can adjust
# MAGIC             minor_oversampled = minor_df.withColumn('dummy', explode(array([lit(x) for x in range(ratio)]))).drop('dummy')
# MAGIC             return major_df.union(minor_oversampled)
# MAGIC 
# MAGIC         def undersample_majority_class(df):
# MAGIC             minor_df = df.where(df[self.configuration.label_col] == 1)
# MAGIC             major_df = df.where(df[self.configuration.label_col] == 0)
# MAGIC             # ratio = int(negatives/positives) # defined in source
# MAGIC             ratio = 3 # for experimentation, can adjust
# MAGIC             major_undersampled = major_df.sample(False, 1/ratio)
# MAGIC             return major_undersampled.union(minor_df)
# MAGIC 
# MAGIC 
# MAGIC         if (apply_class_weights):
# MAGIC             return add_class_weights(df)
# MAGIC         else:
# MAGIC             return undersample_majority_class(df)
# MAGIC 
# MAGIC class CVStrategies():
# MAGIC     def version():
# MAGIC         return '5.0'
# MAGIC     
# MAGIC     def __init__(self):
# MAGIC         self.no_cv = 'None'
# MAGIC         self.rolling_window = 'Rolling Window'
# MAGIC         self.block_quarters = 'Block Quarters'
# MAGIC         self.block_years = 'Block Years'
# MAGIC 
# MAGIC class CVSplitter():
# MAGIC     def version():
# MAGIC         return '1.0'
# MAGIC 
# MAGIC     def CV_noCV(df):
# MAGIC         d = {}
# MAGIC 
# MAGIC         d['df1'] = df.filter(df.YEAR <= 2018)\
# MAGIC                            .withColumn('cv', F.when(df.YEAR <= 2017, 'train')
# MAGIC                                                  .otherwise('test'))
# MAGIC         return d
# MAGIC 
# MAGIC     def CV_rolling_window(df):
# MAGIC         d = {}
# MAGIC 
# MAGIC         d['df1'] = df.filter(df.YEAR <= 2016)\
# MAGIC                            .withColumn('cv', F.when(df.YEAR == 2015, 'train')
# MAGIC                                                  .otherwise('test'))
# MAGIC 
# MAGIC         d['df2'] = df.filter(df.YEAR <= 2017)\
# MAGIC                            .withColumn('cv', F.when(df.YEAR <= 2016, 'train')
# MAGIC                                                  .otherwise('test'))
# MAGIC 
# MAGIC         d['df3'] = df.filter(df.YEAR <= 2018)\
# MAGIC                            .withColumn('cv', F.when(df.YEAR <= 2017, 'train')
# MAGIC                                                  .otherwise('test'))
# MAGIC         return d
# MAGIC 
# MAGIC     def CV_block_by_year(df):
# MAGIC         d= {}
# MAGIC         
# MAGIC         d['df1'] = df.filter(df.YEAR.isin(2015,2016)) \
# MAGIC                             .withColumn('cv', F.when(df.YEAR == 2015, 'train')
# MAGIC                                                      .otherwise('test'))
# MAGIC         d['df2'] = df.filter(df.YEAR.isin(2017,2018)) \
# MAGIC                             .withColumn('cv', F.when(df.YEAR == 2017, 'train')
# MAGIC                                                      .otherwise('test'))
# MAGIC         return d
# MAGIC         
# MAGIC     def CV_block_by_quarter(df):
# MAGIC         d = {}
# MAGIC 
# MAGIC         d['df1'] = df.filter((df.YEAR.isin(2015,2016)) & (df.QUARTER == 1))\
# MAGIC                            .withColumn('cv', F.when((df.YEAR == 2015), 'train')
# MAGIC                                                  .otherwise('test'))
# MAGIC 
# MAGIC         d['df2'] = df.filter((df.YEAR.isin(2015,2016)) & (df.QUARTER == 3))\
# MAGIC                            .withColumn('cv', F.when((df.YEAR == 2015), 'train')
# MAGIC                                                  .otherwise('test'))
# MAGIC 
# MAGIC         d['df3'] = df.filter((df.YEAR.isin(2016,2017)) & (df.QUARTER == 2))\
# MAGIC                            .withColumn('cv', F.when((df.YEAR == 2016), 'train')
# MAGIC                                                  .otherwise('test'))
# MAGIC 
# MAGIC         d['df4'] = df.filter((df.YEAR.isin(2016,2017)) & (df.QUARTER == 4))\
# MAGIC                            .withColumn('cv', F.when((df.YEAR == 2016), 'train')
# MAGIC                                                  .otherwise('test'))
# MAGIC 
# MAGIC         d['df5'] = df.filter((df.YEAR.isin(2017,2018)) & (df.QUARTER == 1))\
# MAGIC                            .withColumn('cv', F.when((df.YEAR == 2017), 'train')
# MAGIC                                                  .otherwise('test'))
# MAGIC 
# MAGIC         d['df6'] = df.filter((df.YEAR.isin(2017,2018)) & (df.QUARTER == 3))\
# MAGIC                            .withColumn('cv', F.when((df.YEAR == 2017), 'train')
# MAGIC                                                  .otherwise('test'))
# MAGIC 
# MAGIC         d['df7'] = df.filter((df.YEAR.isin(2015,2018)) & (df.QUARTER == 2))\
# MAGIC                            .withColumn('cv', F.when((df.YEAR == 2015), 'train')
# MAGIC                                                  .otherwise('test'))
# MAGIC 
# MAGIC         d['df8'] = df.filter((df.YEAR.isin(2015,2018)) & (df.QUARTER == 4))\
# MAGIC                            .withColumn('cv', F.when((df.YEAR == 2015), 'train')
# MAGIC                                                  .otherwise('test'))
# MAGIC         return d

# COMMAND ----------

ls /dbfs/user/ram.senth@berkeley.edu/tmp

# COMMAND ----------

# mv fbeta_evaluator.py /dbfs/user/ram.senth@berkeley.edu/fbeta_evaluator.py
dbutils.fs.mv(f'dbfs:/user/ram.senth@berkeley.edu/tmp/{filename}', f'{configuration.MOUNTED_BLOB_STORE}/library/{filename}')

# COMMAND ----------

# MAGIC %%sh
# MAGIC cat /dbfs/mnt/team11-blobstore/library/data_preparer_v06.py

# COMMAND ----------


