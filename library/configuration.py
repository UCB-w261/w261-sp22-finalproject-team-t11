# Databricks notebook source
# MAGIC %md # Configuration

# COMMAND ----------

MOUNTED_BLOB_STORE = '/mnt/team11-blobstore'
filename = 'configuration_v01.py'

# COMMAND ----------

# MAGIC %%writefile /dbfs/user/ram.senth@berkeley.edu/tmp/configuration_v01.py
# MAGIC class DataSet():
# MAGIC     def version():
# MAGIC         return '1.0'
# MAGIC     
# MAGIC     def __init__(self, training_dataset, training_dataset_name, test_dataset, test_dataset_name):
# MAGIC         self.training_dataset = training_dataset
# MAGIC         self.training_dataset_name = training_dataset_name
# MAGIC         self.test_dataset = test_dataset
# MAGIC         self.test_dataset_name = test_dataset_name        
# MAGIC         
# MAGIC class Configuration():
# MAGIC     def version():
# MAGIC         return '1.0'
# MAGIC     
# MAGIC     def __init__(self):
# MAGIC         self.blob_container = "w261team11" # The name of your container created in https://portal.azure.com
# MAGIC         self.storage_account = "w261sa" # The name of your Storage account created in https://portal.azure.com
# MAGIC         self.secret_scope = "w261team11" # The name of the scope created in your local computer using the Databricks CLI
# MAGIC         self.secret_key = "w261team11key" # The name of the secret key created in your local computer using the Databricks CLI 
# MAGIC         self.blob_url = f"wasbs://{self.blob_container}@{self.storage_account}.blob.core.windows.net"
# MAGIC         self.MOUNTED_BLOB_STORE = '/mnt/team11-blobstore'
# MAGIC         
# MAGIC         self.feature_cols = ['origin_weather_Avg_Elevation',
# MAGIC                 'origin_weather_Avg_HourlyDryBulbTemperature',
# MAGIC                 'origin_weather_Avg_HourlyWindSpeed',
# MAGIC                 'origin_weather_NonZero_Rain',
# MAGIC                 'origin_weather_HourlyPressureTendency_Increasing',
# MAGIC                 'origin_weather_Present_Weather_Hail', 
# MAGIC                 'origin_weather_Present_Weather_Storm', 
# MAGIC                 'dest_weather_Avg_Elevation',
# MAGIC                 'dest_weather_Avg_HourlyDryBulbTemperature',
# MAGIC                 'dest_weather_Avg_HourlyWindSpeed',
# MAGIC                 'dest_weather_NonZero_Rain',
# MAGIC                 'dest_weather_HourlyPressureTendency_Increasing',
# MAGIC                 'dest_weather_Present_Weather_Hail',
# MAGIC                 'dest_weather_Present_Weather_Storm', 
# MAGIC                 'prior_delayed', 
# MAGIC                 '_QUARTER', 
# MAGIC                 '_MONTH', 
# MAGIC                 '_DAY_OF_WEEK', 
# MAGIC                 '_CRS_DEPT_HR', 
# MAGIC                 '_DISTANCE_GROUP', 
# MAGIC                 '_AIRLINE_DELAYS', 
# MAGIC                 '_origin_airport_type', 
# MAGIC                 '_dest_airport_type',
# MAGIC                 'Holiday_5Day']
# MAGIC 
# MAGIC         self.orig_label_col = 'DEP_DEL15'
# MAGIC         self.label_col = 'label'
# MAGIC         self.MODEL_PLOTS_BASE_PATH = '/dbfs/FileStore/shared_uploads/ram.senth@berkeley.edu/model_plots'
# MAGIC         self.TRAINED_MODEL_BASE_PATH = f'{self.blob_url}/model/trained'
# MAGIC         self.MODEL_HISTORY_PATH = f'{self.TRAINED_MODEL_BASE_PATH}/history'
# MAGIC         self.CHECKPOINT_FOLDER = f'{self.blob_url}/checkpoints'
# MAGIC         self.MODEL_NAME = 'lr_with_fbeta'
# MAGIC         self.class_weights_col = 'classWeights'
# MAGIC         self.features_col = 'features'
# MAGIC 
# MAGIC         # Locations of final joined data.
# MAGIC         self.FINAL_JOINED_DATA_2015_2018 = f"{self.blob_url}/staged/final_joined_training"
# MAGIC         self.FINAL_JOINED_DATA_2019 = f"{self.blob_url}/staged/final_joined_testing"
# MAGIC         self.FINAL_JOINED_DATA_2020_2021 = f"{self.blob_url}/staged/final_joined_2020_2021"
# MAGIC         self.FINAL_JOINED_DATA_2020 = f"{self.blob_url}/staged/final_joined_2020"
# MAGIC         self.FINAL_JOINED_DATA_2021 = f"{self.blob_url}/staged/final_joined_2021"
# MAGIC         self.FINAL_JOINED_DATA_Q1_2015_2018 = f"{self.blob_url}/staged/final_joined_Q1_2015_2018"
# MAGIC         self.FINAL_JOINED_DATA_Q1_2019 = f"{self.blob_url}/staged/final_joined_Q1_2019"
# MAGIC 
# MAGIC         #Location for transformed data, ready for training
# MAGIC         self.TRANSFORMED_TRAINING_DATA = f"{self.blob_url}/transformed/training"
# MAGIC         self.TRANSFORMED_2019_DATA = f"{self.blob_url}/transformed/2019"
# MAGIC         self.TRANSFORMED_2020_DATA = f"{self.blob_url}/transformed/2020"
# MAGIC         self.TRANSFORMED_2021_DATA = f"{self.blob_url}/transformed/2021"
# MAGIC         # TRANSFORMED_2020_2021_DATA = f"{self.blob_url}/transformed/2020-2021"
# MAGIC         self.TRANSFORMED_Q1_2015_2018_DATA = f"{self.blob_url}/transformed/Q1-2015-2018"
# MAGIC         self.TRANSFORMED_Q1_2019_DATA = f"{self.blob_url}/transformed/Q1-2019"
# MAGIC 
# MAGIC         # Locations for StringIndexer and OneHotEncoder Models
# MAGIC         self.SI_MODEL_LOC = f"{self.blob_url}/model/train_string_indexer_model"
# MAGIC         self.OHE_MODEL_LOC = f"{self.blob_url}/model/train_one_hot_encoder_model"
# MAGIC         self.SCALER_MODEL_LOC = f"{self.blob_url}/model/train_scaler_encoder_model"
# MAGIC         self.DATA_AUDIT_LOC = f'{self.blob_url}/data_audit'
# MAGIC 
# MAGIC         # Predefined datasets.
# MAGIC         #Full dataset.
# MAGIC         self.FULL_DATASET = DataSet(self.TRANSFORMED_TRAINING_DATA, '2015-2018', self.TRANSFORMED_2019_DATA, '2019')
# MAGIC 
# MAGIC         # Toy dataset
# MAGIC         self.TOY_DATASET = DataSet(self.TRANSFORMED_Q1_2015_2018_DATA, 'Q1 2015-2018', self.TRANSFORMED_Q1_2019_DATA, 'Q1 2019')

# COMMAND ----------

ls /dbfs/user/ram.senth@berkeley.edu/tmp

# COMMAND ----------

dbutils.fs.mv(f'dbfs:/user/ram.senth@berkeley.edu/tmp/{filename}', f'{MOUNTED_BLOB_STORE}/library/{filename}')

# COMMAND ----------

# MAGIC %%sh
# MAGIC cat /dbfs/mnt/team11-blobstore/library/configuration_v01.py

# COMMAND ----------

# MAGIC %%sh
# MAGIC ls /dbfs/mnt/team11-blobstore/library/

# COMMAND ----------


