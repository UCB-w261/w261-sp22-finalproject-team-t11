# Databricks notebook source
# MAGIC %md #Custom Evaluator Using FBeta Score

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

MOUNTED_BLOB_STORE = '/mnt/team11-blobstore'

# COMMAND ----------

def file_exists(path):
      try:
        dbutils.fs.ls(path)
        return True
      except Exception as e:
        if 'java.io.FileNotFoundException' in str(e):
            return False
        else:
            raise

def unmount_blob_store(mount_point):
    if file_exists(mount_point):
        dbutils.fs.unmount(mount_point)

def mount_blob_store(mount_point):
    if not file_exists(mount_point):
        conf_key = f'fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net'
        dbutils.fs.mount(
          source = blob_url,
          mount_point = mount_point,
          extra_configs = {conf_key:dbutils.secrets.get(scope = secret_scope, key = secret_key)})

# COMMAND ----------

# MAGIC %%writefile fbeta_evaluator.py
# MAGIC 
# MAGIC import random
# MAGIC from pyspark.ml.evaluation import Evaluator
# MAGIC 
# MAGIC class FBetaEvaluator(Evaluator):
# MAGIC 
# MAGIC     def __init__(self, predictionCol="prediction", labelCol="label", beta=1.0, labelWeighCol=None):
# MAGIC         self.predictionCol = predictionCol
# MAGIC         self.labelCol = labelCol
# MAGIC         self.beta = beta
# MAGIC 
# MAGIC     def _evaluate(self, dataset):
# MAGIC         # TODO: the col names are hardcoded. Use the parameter instead.
# MAGIC         TP = dataset.filter((dataset.label == 1) & (dataset.prediction == 1)).count()
# MAGIC         TN = dataset.filter((dataset.label == 0) & (dataset.prediction == 0)).count()
# MAGIC         FP = dataset.filter((dataset.label == 0) & (dataset.prediction == 1)).count()
# MAGIC         FN = dataset.filter((dataset.label == 1) & (dataset.prediction == 0)).count()
# MAGIC         
# MAGIC         if (TP+TN+FP+FN) != 0:
# MAGIC             accuracy = (TP + TN) / (TP+TN+FP+FN)
# MAGIC         else:
# MAGIC             accuracy = 0
# MAGIC         if (TP + FP) != 0:
# MAGIC             precision = TP / (TP + FP)
# MAGIC         else:
# MAGIC             precision = 0
# MAGIC         if (TP + FN) != 0:
# MAGIC             recall = TP / (TP + FN)
# MAGIC         else:
# MAGIC             recall = 0
# MAGIC         if ((self.beta**2) * precision + recall) != 0:
# MAGIC             f_beta = ((1+self.beta**2) * (precision * recall)) / ((self.beta**2) * precision + recall)
# MAGIC         else:
# MAGIC             f_beta = 0
# MAGIC         if (precision + recall) != 0:
# MAGIC             f_score = (precision * recall) / (precision + recall)
# MAGIC         else:
# MAGIC             f_score = 0
# MAGIC         
# MAGIC         return f_beta
# MAGIC 
# MAGIC     def isLargerBetter(self):
# MAGIC         return True
# MAGIC     
# MAGIC     def getMetricName(self):
# MAGIC         return 'FBeta'

# COMMAND ----------

mv fbeta_evaluator.py /dbfs/user/ram.senth@berkeley.edu/fbeta_evaluator.py

# COMMAND ----------

ls /dbfs/user/ram.senth@berkeley.edu/

# COMMAND ----------

# MAGIC %%sh
# MAGIC cat /dbfs/user/ram.senth@berkeley.edu/fbeta_evaluator.py

# COMMAND ----------


    

# COMMAND ----------

mount_blob_store(MOUNTED_BLOB_STORE)

# COMMAND ----------

unmount_blob_store(MOUNTED_BLOB_STORE)

# COMMAND ----------

dbutils.fs.ls('/mnt')
dbutils.fs.

# COMMAND ----------


