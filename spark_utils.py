"""
Spark utilities for MySQL connection
"""
from pyspark.sql import SparkSession
from config import SPARK_CONFIG, DATABASE_CONFIG

def create_spark_session_with_mysql():
    """Tạo Spark session với MySQL connector"""
    mysql_jar_path = SPARK_CONFIG["processing"]["configs"]["spark.jars"]
    
    spark = SparkSession.builder \
        .appName(SPARK_CONFIG["processing"]["app_name"]) \
        .config("spark.jars", mysql_jar_path) \
        .config("spark.driver.extraClassPath", mysql_jar_path) \
        .config("spark.executor.extraClassPath", mysql_jar_path) \
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
        .getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

def create_spark_session_for_preprocessing():
    """Tạo Spark session cho preprocessing"""
    spark = SparkSession.builder \
        .appName(SPARK_CONFIG["preprocessing"]["app_name"]) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    return spark
