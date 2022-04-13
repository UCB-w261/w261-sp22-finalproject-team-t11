# Databricks notebook source
filename = 'down_sampler_v02.py'
from configuration_v01 import Configuration
configuration = Configuration()


# COMMAND ----------

# MAGIC %%writefile /dbfs/user/ram.senth@berkeley.edu/tmp/down_sampler_v02.py
# MAGIC 
# MAGIC from pyspark import keyword_only
# MAGIC from pyspark.ml import Transformer
# MAGIC from pyspark.ml.param.shared import HasInputCols, HasOutputCol, HasLabelCol, Param, Params, TypeConverters
# MAGIC from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable  
# MAGIC from pyspark.ml.feature import VectorAssembler
# MAGIC from pyspark.ml.functions import vector_to_array
# MAGIC 
# MAGIC # Credits https://stackoverflow.com/a/52467470
# MAGIC # by https://stackoverflow.com/users/234944/benjamin-manns
# MAGIC class DownSampler(
# MAGIC         Transformer, HasLabelCol, DefaultParamsReadable, DefaultParamsWritable):
# MAGIC 
# MAGIC     ratio = Param(Params._dummy(), "ratio", "ratio",
# MAGIC                       typeConverter=TypeConverters.toFloat)
# MAGIC 
# MAGIC     def version():
# MAGIC         return '2.0'
# MAGIC 
# MAGIC     @keyword_only
# MAGIC     def __init__(self, labelCol=None, ratio=None):
# MAGIC         super(DownSampler, self).__init__()
# MAGIC         
# MAGIC         self.ratio = Param(self, "ratio", "")
# MAGIC         self._setDefault(ratio=3.0)
# MAGIC         
# MAGIC         self._setDefault(labelCol='DEP_DEL15')
# MAGIC         
# MAGIC         kwargs = self._input_kwargs
# MAGIC         self.setParams(**kwargs)
# MAGIC 
# MAGIC     @keyword_only
# MAGIC     def setParams(self, labelCol=None, ratio=None):
# MAGIC         kwargs = self._input_kwargs
# MAGIC         return self._set(**kwargs)
# MAGIC 
# MAGIC     def setRatio(self, value):
# MAGIC         return self._set(ratio=float(value))
# MAGIC 
# MAGIC     def getRatio(self):
# MAGIC         return self.getOrDefault(self.ratio)
# MAGIC     
# MAGIC     # Required in Spark >= 3.0
# MAGIC     def setLabelCol(self, value):
# MAGIC         """
# MAGIC         Sets the value of :py:attr:`labelCol`.
# MAGIC         """
# MAGIC         return self._set(labelCol=value)
# MAGIC 
# MAGIC     def _transform(self, dataset):
# MAGIC         ratio = self.getRatio()
# MAGIC         label_col = self.getLabelCol()
# MAGIC         # TODO: Assumes class 0 is the major class. Can be made more dynamic by comparing counts.
# MAGIC         minor_df = dataset.where(dataset[label_col] == 1)
# MAGIC         major_df = dataset.where(dataset[label_col] == 0)
# MAGIC         # TODO: blindly picks one-third of the major class. Can be smarter.
# MAGIC         major_undersampled = major_df.sample(replace=False, frac=1/ratio)
# MAGIC         transformed = major_undersampled.union(minor_df)
# MAGIC         return transformed

# COMMAND ----------

ls /dbfs/user/ram.senth@berkeley.edu/tmp

# COMMAND ----------

dbutils.fs.mv(f'dbfs:/user/ram.senth@berkeley.edu/tmp/{filename}', f'{configuration.MOUNTED_BLOB_STORE}/library/{filename}')

# COMMAND ----------


