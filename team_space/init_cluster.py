# Databricks notebook source
# MAGIC %md #Run Once After Each Cluster Restart
# MAGIC Since the spark context is created just once in the beginning and is used in all notebooks, we have to set it all up in just one place.

# COMMAND ----------

# Setup Blob store access
blob_container = "w261team11" # The name of your container created in https://portal.azure.com
storage_account = "w261sa" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261team11" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261team11key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"

# Setup spark context.
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# Load libraries stored on DBFS.
spark.sparkContext.addPyFile("dbfs:/user/ram.senth@berkeley.edu/utility_01.py")
from utility_01 import Utility
spark.sparkContext.addPyFile("dbfs:/user/ram.senth@berkeley.edu/curve_metrics.py")
from curve_metrics import CurveMetrics
spark.sparkContext.addPyFile("dbfs:/custom_cv.py")
from custom_cv import CustomCrossValidator
spark.sparkContext.addPyFile("dbfs:/user/ram.senth@berkeley.edu/fbeta_evaluator.py")
from fbeta_evaluator import FBetaEvaluator

# Mount blob store to DBFS.
MOUNTED_BLOB_STORE = '/mnt/team11-blobstore'
Utility.mount_blob_store(dbutils, MOUNTED_BLOB_STORE)

# Load libraries published to blobstore.
spark.sparkContext.addPyFile(f'{blob_url}/library/configuration_v01.py')
from configuration_v01 import Configuration
spark.sparkContext.addPyFile(f'{blob_url}/library/data_preparer_v05.py')
spark.sparkContext.addPyFile(f'{blob_url}/library/data_preparer_v06.py')
from data_preparer_v06 import DataPreparer
spark.sparkContext.addPyFile(f'{blob_url}/library/training_summarizer_v01.py')
from training_summarizer_v01 import TrainingSummarizer

spark.sparkContext.addPyFile(f'{blob_url}/library/down_sampler_v02.py')
from down_sampler_v02 import DownSampler

spark.sparkContext.addPyFile(f'{blob_url}/library/custom_transformer_v05.py')
from custom_transformer_v05 import CustomTransformer

configuration = Configuration()
spark.sparkContext.setCheckpointDir(configuration.CHECKPOINT_FOLDER)

print(f'Configuration.version: {Configuration.version()}')
print(f'DataPreparer version: {DataPreparer.version()}')
print(f'Utility version: {Utility.version()}')
print(f'TrainingSummarizer version: {TrainingSummarizer.version()}')


# COMMAND ----------


