from spark_utils import create_spark_session_with_mysql
from config import DATABASE_CONFIG, ANALYSIS_TABLES, FILE_PATHS
from pyspark.sql.functions import lit, avg, col, count
import datetime


# Tạo Spark session với MySQL connector
spark = create_spark_session_with_mysql()

# Kết nối MySQL từ config
mysql_url = DATABASE_CONFIG["mysql_url"]
mysql_props = DATABASE_CONFIG["mysql_props"]
hdfs_path = FILE_PATHS["hdfs_raw"]

# Đọc dữ liệu đã preprocessed
# Ưu tiên đọc từ HDFS nếu có, ngược lại từ local
df = None
# đọc từ thư mục
if hdfs_path:
    try:
        df = spark.read.csv(hdfs_path, header=True, inferSchema=True)
        print(f"Đã đọc dữ liệu từ HDFS: {hdfs_path}")
    except Exception as e:
        print(f"Lỗi khi đọc từ HDFS: {e}. Sẽ thử đọc từ local.")
else:
    print("Không có đường dẫn HDFS, sẽ đọc từ local.")
    df = spark.read.csv("file:///home/phamp/Train_cleaned.csv", header=True, inferSchema=True)
df.printSchema()
df.show(5)


from pyspark.sql.functions import avg, col, lit

# ============================================================================
# 1. TỔNG QUAN HIỆU SUẤT GIAO HÀNG - OVERALL PERFORMANCE
# ============================================================================
print("📊 1. PHÂN TÍCH TỔNG QUAN HIỆU SUẤT GIAO HÀNG")
avg_df = df.select(avg(col("`Reached.on.Time_Y.N`")).alias("on_time_rate"))
result_df = avg_df.withColumn("late_rate", lit(1) - col("on_time_rate"))
#thêm tổng đơn
result_df = result_df.withColumn("total_orders", lit(df.count()))

result_df.show()

# Ghi vào MySQL
result_df.write.jdbc(url=mysql_url, table=ANALYSIS_TABLES["overall_performance"],
                    mode="overwrite", properties=mysql_props)

# ============================================================================
# 2. PHÂN TÍCH THEO KHU VỰC KHO - WAREHOUSE BLOCK ANALYSIS
# ============================================================================
print("🏢 2. PHÂN TÍCH THEO KHU VỰC KHO")
warehouse_df = df.groupBy("Warehouse_block") \
	.agg(
        avg(col("`Reached.on.Time_Y.N`")).alias("on_time_rate"),
        count("*").alias("count")
    ) \
	.withColumn("late_rate", 1 - col("on_time_rate")) \
	.orderBy("late_rate", ascending=False)
warehouse_df.show()

# Ghi vào MySQL
warehouse_df.write.jdbc(url=mysql_url, table=ANALYSIS_TABLES["warehouse_analysis"], 
                        mode="overwrite", properties=mysql_props)

# ============================================================================
# 3. PHÂN TÍCH THEO PHƯƠNG THỨC VẬN CHUYỂN - SHIPPING MODE ANALYSIS
# ============================================================================
print("🚚 3. PHÂN TÍCH THEO PHƯƠNG THỨC VẬN CHUYỂN")
shipping_mode_df = df.groupBy("Mode_of_Shipment") \
	.agg(
        avg(col("`Reached.on.Time_Y.N`")).alias("on_time_rate"),
        count("*").alias("count")
    ) \
	.withColumn("late_rate", 1 - col("on_time_rate")) \
	.orderBy("late_rate", ascending=False)
shipping_mode_df.show()

# Ghi vào MySQL
shipping_mode_df.write.jdbc(url=mysql_url, table=ANALYSIS_TABLES["shipping_mode_analysis"], 
                            mode="overwrite", properties=mysql_props)

# ============================================================================
# 4. PHÂN TÍCH THEO ĐỘ QUAN TRỌNG SẢN PHẨM - PRODUCT IMPORTANCE ANALYSIS
# ============================================================================
print("⭐ 4. PHÂN TÍCH THEO ĐỘ QUAN TRỌNG SẢN PHẨM")
product_importance_df = df.groupBy("Product_importance") \
	.agg(
        avg(col("`Reached.on.Time_Y.N`")).alias("on_time_rate"),
        count("*").alias("count")
    ) \
	.withColumn("late_rate", 1 - col("on_time_rate")) \
	.orderBy("late_rate", ascending=False)
product_importance_df.show()

# Ghi vào MySQL
product_importance_df.write.jdbc(url=mysql_url, table=ANALYSIS_TABLES["product_importance_analysis"], 
                                mode="overwrite", properties=mysql_props)

from pyspark.sql.functions import when

# ============================================================================
# 5. PHÂN TÍCH THEO MỨC ĐỘ GIẢM GIÁ - DISCOUNT LEVEL ANALYSIS
# ============================================================================
print("💰 5. PHÂN TÍCH THEO MỨC ĐỘ GIẢM GIÁ")
# Tạo categories: Low (<=10%), Medium (10-30%), High (>30%)
df = df.withColumn("Discount_level",
    when(col("Discount_offered") <= 7, "Low")
    .when((col("Discount_offered") > 7) & (col("Discount_offered") <= 20.5), "Medium")
    .otherwise("High")
)
discount_df = df.groupBy("Discount_level") \
	.agg(
		avg(col("`Reached.on.Time_Y.N`")).alias("on_time_rate"),
		count("*").alias("count")
	) \
	.withColumn("late_rate", 1 - col("on_time_rate")) \
	.orderBy("late_rate", ascending=False)
discount_df.show()

# Ghi vào MySQL
discount_df.write.jdbc(url=mysql_url, table=ANALYSIS_TABLES["discount_level_analysis"], 
                    mode="overwrite", properties=mysql_props)

# ============================================================================
# 6. PHÂN TÍCH THEO DANH MỤC TRỌNG LƯỢNG - WEIGHT CATEGORY ANALYSIS
# ============================================================================
print("⚖️ 6. PHÂN TÍCH THEO DANH MỤC TRỌNG LƯỢNG")
# Tạo categories: Light (<=2000g), Medium (2000-4000g), Heavy (>4000g)

df = df.withColumn("Weight_category",
    when(col("Weight_in_gms") <= 3000, "Light")
    .when((col("Weight_in_gms") > 3000) & (col("Weight_in_gms") <= 5500), "Medium")
    .otherwise("Heavy")
)
#in số lượng mỗi phân loại
weight_category_counts = df.groupBy("Weight_category").agg(count("*").alias("count"))
weight_category_counts.show()


weight_df = df.groupBy("Weight_category") \
	.agg(avg(col("`Reached.on.Time_Y.N`")).alias("on_time_rate"),
		count("*").alias("count")
	) \
	.withColumn("late_rate", 1 - col("on_time_rate")) \
	.orderBy("late_rate", ascending=False)
weight_df.show()

# Ghi vào MySQL
weight_df.write.jdbc(url=mysql_url, table=ANALYSIS_TABLES["weight_category_analysis"], 
                    mode="overwrite", properties=mysql_props)

# ============================================================================
# 7. PHÂN TÍCH KẾT HỢP - SHIPPING MODE & WEIGHT CATEGORY ANALYSIS
# ============================================================================
print("🚛 7. PHÂN TÍCH KẾT HỢP: PHƯƠNG THỨC VẬN CHUYỂN & TRỌNG LƯỢNG")
# Phân tích cross-dimensional: Phương thức vận chuyển × Trọng lượng
combined_df = df.groupBy("Mode_of_Shipment", "Weight_category") \
	.agg(avg(col("`Reached.on.Time_Y.N`")).alias("on_time_rate"),
		count("*").alias("count")
	) \
	.withColumn("late_rate", 1 - col("on_time_rate")) \
	.orderBy("late_rate", ascending=False)
combined_df.show()

# Ghi vào MySQL
combined_df.write.jdbc(url=mysql_url, table=ANALYSIS_TABLES["combined_mode_weight"], 
                       mode="overwrite", properties=mysql_props)

# ============================================================================
# 8. PHÂN TÍCH THEO ĐÁNH GIÁ KHÁCH HÀNG - CUSTOMER RATING ANALYSIS
# ============================================================================
print("⭐ 8. PHÂN TÍCH THEO ĐÁNH GIÁ KHÁCH HÀNG")
# Phân tích mối quan hệ giữa đánh giá khách hàng và hiệu suất giao hàng
rating_df = df.groupBy("Customer_rating") \
	.agg(avg(col("`Reached.on.Time_Y.N`")).alias("on_time_rate"),
		count("*").alias("count")
	) \
	.withColumn("late_rate", 1 - col("on_time_rate")) \
	.orderBy("Customer_rating")
rating_df.show()

# Ghi vào MySQL
rating_df.write.jdbc(url=mysql_url, table=ANALYSIS_TABLES["customer_rating_analysis"], 
                    mode="overwrite", properties=mysql_props)

# ============================================================================
# 9. PHÂN TÍCH THEO SỐ CUỘC GỌI CHĂM SÓC - CUSTOMER CARE CALLS ANALYSIS
# ============================================================================
print("📞 9. PHÂN TÍCH THEO SỐ CUỘC GỌI CHĂM SÓC KHÁCH HÀNG")
# Phân tích tác động của số cuộc gọi chăm sóc khách hàng đến hiệu suất giao hàng
care_calls_df = df.groupBy("Customer_care_calls") \
	.agg(avg(col("`Reached.on.Time_Y.N`")).alias("on_time_rate"),
		count("*").alias("count")
	) \
	.withColumn("late_rate", 1 - col("on_time_rate")) \
	.orderBy("Customer_care_calls")
care_calls_df.show()

# Ghi vào MySQL
care_calls_df.write.jdbc(url=mysql_url, table=ANALYSIS_TABLES["customer_care_analysis"], 
                        mode="overwrite", properties=mysql_props)

# ============================================================================
# 10. PHÂN TÍCH THEO LỊCH SỬ MUA HÀNG - PRIOR PURCHASES ANALYSIS
# ============================================================================
print("🛒 10. PHÂN TÍCH THEO LỊCH SỬ MUA HÀNG")
# Phân tích mối quan hệ giữa lịch sử mua hàng và hiệu suất giao hàng
purchases_df = df.groupBy("Prior_purchases") \
	.agg(avg(col("`Reached.on.Time_Y.N`")).alias("on_time_rate"),
		count("*").alias("count")
	) \
	.withColumn("late_rate", 1 - col("on_time_rate")) \
	.orderBy("Prior_purchases")
purchases_df.show()

# Ghi vào MySQL
purchases_df.write.jdbc(url=mysql_url, table=ANALYSIS_TABLES["prior_purchases_analysis"], 
                        mode="overwrite", properties=mysql_props)

print("✅ Đã hoàn thành tất cả 10 phân tích và ghi dữ liệu vào MySQL!")
spark.stop()
