from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, concat_ws, lit, current_timestamp, sum, max, min, stddev
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import datetime
from analysis.config import DATABASE_CONFIG, FILE_PATHS, ANALYSIS_TABLES
from spark_utils import create_spark_session_with_mysql

# Tạo Spark session với MySQL connector
spark = create_spark_session_with_mysql()

# Kết nối MySQL từ config
mysql_url = DATABASE_CONFIG["mysql_url"]
mysql_props = DATABASE_CONFIG["mysql_props"]

# Đọc dữ liệu đã preprocessed
# Ưu tiên đọc từ HDFS nếu có, ngược lại từ local
df = None
try:
    df = spark.read.csv(FILE_PATHS["hdfs_preprocessed"], header=True, inferSchema=True)
    print("Đã đọc dữ liệu preprocessed từ HDFS")
except:
    try:
        df = spark.read.csv(FILE_PATHS["preprocessed_data"] + "/*.csv", header=True, inferSchema=True)
        print("Đã đọc dữ liệu preprocessed từ local")
    except Exception as e:
        print(f"Lỗi khi đọc dữ liệu preprocessed: {e}")
        print("Vui lòng chạy spark_preprocessing.py trước để tạo dữ liệu preprocessed")
        spark.stop()
        exit(1)

# Kiểm tra schema của dữ liệu preprocessed
print("Schema của dữ liệu preprocessed:")
df.printSchema()

# Tạo aliases cho các cột để dễ sử dụng trong analysis (chỉ nếu cần)
# Dữ liệu preprocessed đã có tên cột chuẩn rồi
if "mode_of_shipment" in df.columns:
    df = df.withColumnRenamed("mode_of_shipment", "mode")
if "customer_care_calls" in df.columns:
    df = df.withColumnRenamed("customer_care_calls", "calls")
if "customer_rating" in df.columns:
    df = df.withColumnRenamed("customer_rating", "rating")
if "cost_of_product" in df.columns:
    df = df.withColumnRenamed("cost_of_product", "cost")
if "prior_purchases" in df.columns:
    df = df.withColumnRenamed("prior_purchases", "purchases")
if "product_importance" in df.columns:
    df = df.withColumnRenamed("product_importance", "importance")
if "warehouse_block" in df.columns:
    df = df.withColumnRenamed("warehouse_block", "warehouse")
if "discount_offered" in df.columns:
    df = df.withColumnRenamed("discount_offered", "discount")
if "weight_in_gms" in df.columns:
    df = df.withColumnRenamed("weight_in_gms", "weight")
if "reached_on_time" in df.columns:
    df = df.withColumnRenamed("reached_on_time", "late")

# Thêm timestamp cho lần chạy
timestamp = datetime.datetime.now().isoformat()
df = df.withColumn("batch_time", lit(timestamp))

# Kiểm tra xem có derived features từ preprocessing không
has_derived_features = "cost_per_gram" in df.columns

print(f"Đang xử lý {df.count()} records với timestamp: {timestamp}")
print(f"Có derived features: {has_derived_features}")

# Helper function để write với deduplication
def write_to_mysql(df, table_name, mode="append"):
    """Write DataFrame to MySQL with optional deduplication"""
    try:
        if mode == "overwrite":
            # For summary stats, we want to replace all data
            df.write.jdbc(url=mysql_url, table=table_name, mode="overwrite", properties=mysql_props)
        else:
            # For time-series data, append with timestamp
            df.write.jdbc(url=mysql_url, table=table_name, mode="append", properties=mysql_props)
        print(f"✅ Successfully written to {table_name}")
    except Exception as e:
        print(f"❌ Error writing to {table_name}: {e}")

# 1. Tổng quan hiệu suất (Enhanced)
summary_stats = df.agg(
    count("*").alias("total"),
    count(when(col("late") == 0, True)).alias("on_time"),
    avg("rating").alias("avg_rating"),
    avg("calls").alias("avg_calls"),
    avg("cost").alias("avg_cost"),
    avg("weight").alias("avg_weight"),
    avg("discount").alias("avg_discount"),
    stddev("rating").alias("rating_stddev"),
    min("rating").alias("min_rating"),
    max("rating").alias("max_rating")
).collect()[0]

summary_df = spark.createDataFrame([{
    "batch_time": timestamp,
    "total": summary_stats["total"],
    "on_time": summary_stats["on_time"],
    "on_time_rate": summary_stats["on_time"] / summary_stats["total"] if summary_stats["total"] > 0 else 0,
    "avg_rating": summary_stats["avg_rating"],
    "avg_calls": summary_stats["avg_calls"],
    "avg_cost": summary_stats["avg_cost"],
    "avg_weight": summary_stats["avg_weight"],
    "avg_discount": summary_stats["avg_discount"],
    "rating_stddev": summary_stats["rating_stddev"],
    "min_rating": summary_stats["min_rating"],
    "max_rating": summary_stats["max_rating"]
}])
write_to_mysql(summary_df, ANALYSIS_TABLES["summary"], mode="overwrite")

# 2. Độ trễ theo tuyến
df = df.withColumn("route", concat_ws("_", "warehouse", "mode"))
route_df = df.groupBy("route").agg(
    avg("late").alias("avg_late"),
    avg("cost").alias("avg_cost")
).withColumn("batch_time", lit(timestamp))
write_to_mysql(route_df, ANALYSIS_TABLES["route"])

# 3. Độ trễ theo warehouse
warehouse_df = df.groupBy("warehouse").agg(
    avg("late").alias("avg_late")
).withColumn("batch_time", lit(timestamp))
write_to_mysql(warehouse_df, ANALYSIS_TABLES["warehouse"])

# 4. Độ trễ theo phương thức vận chuyển
mode_df = df.groupBy("mode").agg(
    avg("late").alias("avg_late")
).withColumn("batch_time", lit(timestamp))
write_to_mysql(mode_df, ANALYSIS_TABLES["mode"])

# 5. Trọng lượng và thời gian giao hàng
weight_df = df.groupBy("late").agg(
    avg("weight").alias("avg_weight")
).withColumn("batch_time", lit(timestamp))
write_to_mysql(weight_df, ANALYSIS_TABLES["weight"])

# 6. Phân tích đúng hạn theo warehouse, mode, importance
group_tables = {
    "warehouse": "warehouse_analysis", 
    "mode": "mode_analysis",
    "importance": "importance_analysis"
}

for group_col in ["warehouse", "mode", "importance"]:
    group_df = df.groupBy(group_col).agg(
        count("*").alias("total"),
        count(when(col("late") == 0, True)).alias("on_time"),
        avg("rating").alias("avg_rating"),
        avg("cost").alias("avg_cost"),
        avg("calls").alias("avg_calls")
    ).withColumn("on_time_rate", col("on_time") / col("total")) \
     .withColumn("batch_time", lit(timestamp))
    
    table_name = group_tables[group_col]
    write_to_mysql(group_df, table_name)

# 7. Sản phẩm quan trọng có được đánh giá cao không (Enhanced)
importance_df = df.groupBy("importance").agg(
    count("*").alias("total"),
    avg("rating").alias("avg_rating"),
    avg(when(col("late") == 0, 1).otherwise(0)).alias("on_time_rate"),
    avg("cost").alias("avg_cost"),
    avg("calls").alias("avg_calls"),
    avg("discount").alias("avg_discount")
).withColumn("batch_time", lit(timestamp))
write_to_mysql(importance_df, ANALYSIS_TABLES["importance"])

# 8. Phân tích theo derived features (nếu có)
if has_derived_features:
    # Phân tích theo weight_category
    if "weight_category" in df.columns:
        weight_cat_df = df.groupBy("weight_category").agg(
            count("*").alias("total"),
            avg(when(col("late") == 0, 1).otherwise(0)).alias("on_time_rate"),
            avg("rating").alias("avg_rating"),
            avg("cost").alias("avg_cost"),
            avg("calls").alias("avg_calls")
        ).withColumn("batch_time", lit(timestamp))
        write_to_mysql(weight_cat_df, ANALYSIS_TABLES["weight_category"])
    
    # Phân tích theo cost_category
    if "cost_category" in df.columns:
        cost_cat_df = df.groupBy("cost_category").agg(
            count("*").alias("total"),
            avg(when(col("late") == 0, 1).otherwise(0)).alias("on_time_rate"),
            avg("rating").alias("avg_rating"),
            avg("weight").alias("avg_weight"),
            avg("calls").alias("avg_calls")
        ).withColumn("batch_time", lit(timestamp))
        write_to_mysql(cost_cat_df, ANALYSIS_TABLES["cost_category"])
    
    # Phân tích high_maintenance products
    if "high_maintenance" in df.columns:
        maintenance_df = df.groupBy("high_maintenance").agg(
            count("*").alias("total"),
            avg(when(col("late") == 0, 1).otherwise(0)).alias("on_time_rate"),
            avg("rating").alias("avg_rating"),
            avg("cost").alias("avg_cost"),
            avg("calls").alias("avg_calls")
        ).withColumn("batch_time", lit(timestamp))
        write_to_mysql(maintenance_df, ANALYSIS_TABLES["maintenance"])
    
    # Phân tích cost_per_gram
    if "cost_per_gram" in df.columns:
        cost_per_gram_stats = df.agg(
            avg("cost_per_gram").alias("avg_cost_per_gram"),
            min("cost_per_gram").alias("min_cost_per_gram"),
            max("cost_per_gram").alias("max_cost_per_gram"),
            stddev("cost_per_gram").alias("stddev_cost_per_gram")
        ).withColumn("batch_time", lit(timestamp))
        write_to_mysql(cost_per_gram_stats, ANALYSIS_TABLES["cost_per_gram"])

# 9. Multi-dimensional analysis
# Phân tích kết hợp mode và importance
mode_importance_df = df.groupBy("mode", "importance").agg(
    count("*").alias("total"),
    avg(when(col("late") == 0, 1).otherwise(0)).alias("on_time_rate"),
    avg("rating").alias("avg_rating"),
    avg("cost").alias("avg_cost")
).withColumn("batch_time", lit(timestamp))
mode_importance_df.write.jdbc(url=mysql_url, table=ANALYSIS_TABLES["mode_importance"], mode="append", properties=mysql_props)

# Phân tích kết hợp warehouse và mode
warehouse_mode_df = df.groupBy("warehouse", "mode").agg(
    count("*").alias("total"),
    avg(when(col("late") == 0, 1).otherwise(0)).alias("on_time_rate"),
    avg("rating").alias("avg_rating"),
    avg("cost").alias("avg_cost"),
    avg("weight").alias("avg_weight")
).withColumn("batch_time", lit(timestamp))
warehouse_mode_df.write.jdbc(url=mysql_url, table=ANALYSIS_TABLES["warehouse_mode"], mode="append", properties=mysql_props)

print(f"Hoàn thành phân tích dữ liệu shipping lúc: {timestamp}")
print(f"Đã xử lý tổng cộng {summary_stats['total']} records")
print(f"Tỷ lệ giao hàng đúng hạn: {summary_stats['on_time'] / summary_stats['total'] * 100:.2f}%")

spark.stop()
