# Databricks notebook source
filename = 'custom_transformer_v05.py'
from configuration_v01 import Configuration
configuration = Configuration()


# COMMAND ----------

# MAGIC %%writefile /dbfs/user/ram.senth@berkeley.edu/tmp/custom_transformer_v05.py
# MAGIC 
# MAGIC from pyspark import keyword_only
# MAGIC from pyspark.ml import Transformer
# MAGIC from pyspark.ml.param.shared import HasInputCols, HasOutputCol, HasLabelCol, HasSeed, Param, Params, TypeConverters
# MAGIC from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable  
# MAGIC from pyspark.ml.feature import VectorAssembler
# MAGIC from pyspark.ml.functions import vector_to_array
# MAGIC 
# MAGIC # Credits https://stackoverflow.com/a/52467470
# MAGIC # by https://stackoverflow.com/users/234944/benjamin-manns
# MAGIC class CustomTransformer(
# MAGIC         Transformer, HasInputCols, HasOutputCol, HasLabelCol, HasSeed, DefaultParamsReadable, DefaultParamsWritable):
# MAGIC 
# MAGIC     percentage = Param(Params._dummy(), "percentage", "percentage",
# MAGIC                       typeConverter=TypeConverters.toFloat)
# MAGIC 
# MAGIC     def version():
# MAGIC         return '5.0'
# MAGIC 
# MAGIC     @keyword_only
# MAGIC     def __init__(self, inputCols=None, outputCol=None, labelCol=None, seed=None, percentage=None):
# MAGIC         super(CustomTransformer, self).__init__()
# MAGIC         
# MAGIC         self.percentage = Param(self, "percentage", "")
# MAGIC         self._setDefault(percentage=1.0)
# MAGIC         
# MAGIC         self._setDefault(inputCols = ['DEP_DEL15', 'origin_weather_Present_Weather_Hail', 'origin_weather_Present_Weather_Storm',
# MAGIC         'dest_weather_Present_Weather_Hail', 'dest_weather_Present_Weather_Storm', 'prior_delayed', 'Holiday_5Day'])
# MAGIC         
# MAGIC         self._setDefault(outputCol='features')
# MAGIC         
# MAGIC         self._setDefault(labelCol='DEP_DEL15')
# MAGIC         
# MAGIC         kwargs = self._input_kwargs
# MAGIC         self.setParams(**kwargs)
# MAGIC 
# MAGIC     @keyword_only
# MAGIC     def setParams(self, outputCol=None, inputCols=None, labelCol=None, seed=None, percentage=None):
# MAGIC         kwargs = self._input_kwargs
# MAGIC         return self._set(**kwargs)
# MAGIC 
# MAGIC     def setPercentage(self, value):
# MAGIC         return self._set(stopwords=float(value))
# MAGIC 
# MAGIC     def getPercentage(self):
# MAGIC         return self.getOrDefault(self.percentage)
# MAGIC 
# MAGIC     # Required in Spark >= 3.0
# MAGIC     def setInputCols(self, value):
# MAGIC         return self._set(inputCols=list(value))
# MAGIC 
# MAGIC     # Required in Spark >= 3.0
# MAGIC     def setOutputCol(self, value):
# MAGIC         """
# MAGIC         Sets the value of :py:attr:`outputCol`.
# MAGIC         """
# MAGIC         return self._set(outputCol=value)
# MAGIC     
# MAGIC     # Required in Spark >= 3.0
# MAGIC     def setLabelCol(self, value):
# MAGIC         """
# MAGIC         Sets the value of :py:attr:`labelCol`.
# MAGIC         """
# MAGIC         return self._set(labelCol=value)
# MAGIC 
# MAGIC     # Required in Spark >= 3.0
# MAGIC     def setSeed(self, value):
# MAGIC         return self._set(seed=value)
# MAGIC 
# MAGIC     def _transform(self, dataset):
# MAGIC         percentage = self.getPercentage()
# MAGIC         seed = self.getSeed()
# MAGIC         in_cols = self.getInputCols()
# MAGIC         out_col = self.getOutputCol()
# MAGIC         label_col = self.getLabelCol()
# MAGIC         
# MAGIC         transformed = dataset.sample(False, percentage, seed=seed).select(*in_cols)
# MAGIC         label_removed = list([col for col in in_cols if col != label_col])
# MAGIC         assembler = VectorAssembler(inputCols=label_removed, outputCol=out_col)
# MAGIC         transformed = assembler.transform(transformed).select(*[vector_to_array(out_col).alias(out_col), label_col])
# MAGIC 
# MAGIC         return transformed

# COMMAND ----------

ls /dbfs/user/ram.senth@berkeley.edu/tmp

# COMMAND ----------

dbutils.fs.mv(f'dbfs:/user/ram.senth@berkeley.edu/tmp/{filename}', f'{configuration.MOUNTED_BLOB_STORE}/library/{filename}')

# COMMAND ----------


