# Databricks notebook source
# MAGIC %md #Helper Class For Curve Metrics
# MAGIC 
# MAGIC Reference: https://stackoverflow.com/questions/52847408/pyspark-extract-roc-curve
# MAGIC 
# MAGIC By default the pyspark wrapper of SparkML's BinaryClassificationMetrics does not surface the ROC and AUC metrics. This helper class is a thin layer on top of BinaryClassificationMetrics that pulls both metrics.

# COMMAND ----------

# MAGIC %%writefile curve_metrics.py
# MAGIC 
# MAGIC from pyspark.mllib.evaluation import BinaryClassificationMetrics
# MAGIC 
# MAGIC # Scala version implements .roc() and .pr()
# MAGIC # Python: https://spark.apache.org/docs/latest/api/python/_modules/pyspark/mllib/common.html
# MAGIC # Scala: https://spark.apache.org/docs/latest/api/java/org/apache/spark/mllib/evaluation/BinaryClassificationMetrics.html
# MAGIC class CurveMetrics(BinaryClassificationMetrics):
# MAGIC     def __init__(self, *args):
# MAGIC         super(CurveMetrics, self).__init__(*args)
# MAGIC 
# MAGIC     def _to_list(self, rdd):
# MAGIC         points = []
# MAGIC         # Note this collect could be inefficient for large datasets 
# MAGIC         # considering there may be one probability per datapoint (at most)
# MAGIC         # The Scala version takes a numBins parameter, 
# MAGIC         # but it doesn't seem possible to pass this from Python to Java
# MAGIC         for row in rdd.collect():
# MAGIC             # Results are returned as type scala.Tuple2, 
# MAGIC             # which doesn't appear to have a py4j mapping
# MAGIC             points += [(float(row._1()), float(row._2()))]
# MAGIC         return points
# MAGIC     
# MAGIC     def get_curve(self, method):
# MAGIC         rdd = getattr(self._java_model, method)().toJavaRDD()
# MAGIC         return self._to_list(rdd)

# COMMAND ----------

mv curve_metrics.py /dbfs/user/ram.senth@berkeley.edu/curve_metrics.py

# COMMAND ----------

# MAGIC %%sh
# MAGIC cat /dbfs/user/ram.senth@berkeley.edu/curve_metrics.py

# COMMAND ----------


