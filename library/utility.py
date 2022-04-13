# Databricks notebook source
# MAGIC %md #Utility Methods

# COMMAND ----------

# MAGIC %%writefile /dbfs/user/ram.senth@berkeley.edu/utility_01.py
# MAGIC 
# MAGIC class Utility():
# MAGIC     def version():
# MAGIC         return '1.0'
# MAGIC     
# MAGIC     def file_exists(dbutils, path):
# MAGIC           try:
# MAGIC             dbutils.fs.ls(path)
# MAGIC             return True
# MAGIC           except Exception as e:
# MAGIC             if 'java.io.FileNotFoundException' in str(e):
# MAGIC                 return False
# MAGIC             else:
# MAGIC                 raise
# MAGIC 
# MAGIC     def unmount_blob_store(dbutils, mount_point):
# MAGIC         if Utility.file_exists(dbutils, mount_point):
# MAGIC             dbutils.fs.unmount(mount_point)
# MAGIC 
# MAGIC     def mount_blob_store(dbutils, mount_point):
# MAGIC         if not Utility.file_exists(dbutils, mount_point):
# MAGIC             conf_key = f'fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net'
# MAGIC             dbutils.fs.mount(
# MAGIC               source = blob_url,
# MAGIC               mount_point = mount_point,
# MAGIC               extra_configs = {conf_key:dbutils.secrets.get(scope = secret_scope, key = secret_key)})
# MAGIC     

# COMMAND ----------

# MAGIC %%sh
# MAGIC cat /dbfs/user/ram.senth@berkeley.edu/utility_01.py

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/user/ram.senth@berkeley.edu/

# COMMAND ----------


