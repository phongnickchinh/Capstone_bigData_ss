from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, regexp_replace, trim, lower, upper, split
from pyspark.sql.types import IntegerType, DoubleType, StringType
import os
from analysis.config import SPARK_CONFIG, FILE_PATHS, COLUMN_MAPPINGS, VALIDATION_RULES, FEATURE_ENGINEERING, DATA_QUALITY
from spark_utils import create_spark_session_for_preprocessing

def create_spark_session():
    """Tạo Spark session cho preprocessing"""
    return create_spark_session_for_preprocessing()

def load_raw_data(spark, file_path):
    """Đọc dữ liệu thô từ CSV - Ưu tiên từ HDFS"""
    # Thử đọc từ HDFS trước
    try:
        hdfs_path = FILE_PATHS["hdfs_raw"]
        df = spark.read.csv(hdfs_path, header=True, inferSchema=True)
        print(f"Đã đọc {df.count()} records từ HDFS: {hdfs_path}")
        return df
    except Exception as e:
        print(f"Không thể đọc từ HDFS ({hdfs_path}): {e}")
        
    # Fallback: đọc từ local
    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        print(f"Đã đọc {df.count()} records từ local: {file_path}")
        return df
    except Exception as e:
        print(f"Lỗi khi đọc file từ local: {e}")
        return None

def clean_column_names(df):
    """Làm sạch và chuẩn hóa tên cột"""
    # Sử dụng column mapping từ config
    for old_name, new_name in COLUMN_MAPPINGS.items():
        if old_name in df.columns:
            df = df.withColumnRenamed(old_name, new_name)
    
    return df

def handle_missing_values(df):
    """Xử lý missing values"""
    # Kiểm tra null values
    print("Kiểm tra missing values:")
    for column in df.columns:
        null_count = df.filter(col(column).isNull() | isnan(col(column))).count()
        total_count = df.count()
        if null_count > 0:
            print(f"  {column}: {null_count}/{total_count} ({null_count/total_count*100:.2f}%)")
    
    # Xử lý missing values cho từng cột
    # Fill numerical columns với median hoặc mean
    numerical_cols = ["customer_care_calls", "customer_rating", "cost_of_product", 
                     "prior_purchases", "discount_offered", "weight_in_gms"]
    
    for col_name in numerical_cols:
        if col_name in df.columns:
            # Tính median cho numerical columns
            median_val = df.approxQuantile(col_name, [0.5], 0.01)[0]
            df = df.fillna({col_name: median_val})
    
    # Fill categorical columns với mode hoặc 'Unknown'
    categorical_cols = ["mode_of_shipment", "product_importance", "warehouse_block"]
    
    for col_name in categorical_cols:
        if col_name in df.columns:
            # Tìm mode (giá trị xuất hiện nhiều nhất)
            mode_val = df.groupBy(col_name).count().orderBy(col("count").desc()).first()
            if mode_val:
                mode_value = mode_val[0]
                df = df.fillna({col_name: mode_value})
            else:
                df = df.fillna({col_name: "Unknown"})
    
    return df

def clean_categorical_data(df):
    """Làm sạch dữ liệu categorical"""
    # Chuẩn hóa mode_of_shipment
    if "mode_of_shipment" in df.columns:
        df = df.withColumn("mode_of_shipment", 
                          when(lower(trim(col("mode_of_shipment"))).isin(["ship", "shipping"]), "Ship")
                          .when(lower(trim(col("mode_of_shipment"))).isin(["flight", "air"]), "Flight") 
                          .when(lower(trim(col("mode_of_shipment"))).isin(["road", "truck"]), "Road")
                          .otherwise(upper(trim(col("mode_of_shipment")))))
    
    # Chuẩn hóa product_importance
    if "product_importance" in df.columns:
        df = df.withColumn("product_importance",
                          when(lower(trim(col("product_importance"))).isin(["low"]), "low")
                          .when(lower(trim(col("product_importance"))).isin(["medium"]), "medium")
                          .when(lower(trim(col("product_importance"))).isin(["high"]), "high")
                          .otherwise(lower(trim(col("product_importance")))))
    
    # Chuẩn hóa warehouse_block
    if "warehouse_block" in df.columns:
        df = df.withColumn("warehouse_block", upper(trim(col("warehouse_block"))))
    
    return df

def validate_numerical_data(df):
    """Kiểm tra và làm sạch dữ liệu số"""
    # Sử dụng validation rules từ config
    for col_name, rules in VALIDATION_RULES.items():
        if col_name in df.columns:
            # Áp dụng min constraint nếu có
            if "min" in rules:
                df = df.withColumn(col_name,
                                  when(col(col_name) < rules["min"], rules["default"])
                                  .otherwise(col(col_name)))
            
            # Áp dụng max constraint nếu có
            if "max" in rules:
                df = df.withColumn(col_name,
                                  when(col(col_name) > rules["max"], rules["default"])
                                  .otherwise(col(col_name)))
    
    return df

def create_derived_features(df):
    """Tạo các feature mới từ dữ liệu hiện có"""
    # Tạo cost_per_gram
    if "cost_of_product" in df.columns and "weight_in_gms" in df.columns:
        df = df.withColumn("cost_per_gram", col("cost_of_product") / col("weight_in_gms"))
    
    # Tạo weight_category từ config
    if "weight_in_gms" in df.columns:
        weight_cats = FEATURE_ENGINEERING["weight_categories"]
        df = df.withColumn("weight_category",
                          when(col("weight_in_gms") <= weight_cats["Light"]["max"], "Light")
                          .when(col("weight_in_gms") <= weight_cats["Medium"]["max"], "Medium")
                          .otherwise("Heavy"))
    
    # Tạo cost_category từ config
    if "cost_of_product" in df.columns:
        cost_cats = FEATURE_ENGINEERING["cost_categories"]
        df = df.withColumn("cost_category",
                          when(col("cost_of_product") <= cost_cats["Low"]["max"], "Low")
                          .when(col("cost_of_product") <= cost_cats["Medium"]["max"], "Medium")
                          .otherwise("High"))
    
    # Tạo high_maintenance từ config
    if "customer_care_calls" in df.columns and "product_importance" in df.columns:
        threshold = FEATURE_ENGINEERING["high_maintenance_threshold"]
        df = df.withColumn("high_maintenance",
                          when((col("customer_care_calls") >= threshold["min_calls"]) & 
                               (col("product_importance") == threshold["importance_level"]), 1)
                          .otherwise(0))
    
    return df

def remove_outliers(df):
    """Loại bỏ outliers cho các cột số"""
    numerical_cols = ["customer_care_calls",
                      "customer_rating",
                      "cost_of_product",

                      "weight_in_gms"]
    
    for col_name in numerical_cols:
        if col_name in df.columns:
            # Tính Q1, Q3 và IQR
            quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
            if len(quantiles) == 2:
                Q1, Q3 = quantiles
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # print(f"Loại bỏ outliers cho cột {col_name}: Q1={Q1}, Q3={Q3}, IQR={IQR}, "
                #       f"lower_bound={lower_bound}, upper_bound={upper_bound}")
                
                # Loại bỏ outliers
                initial_count = df.count()
                df = df.filter((col(col_name) >= lower_bound) & (col(col_name) <= upper_bound))
                final_count = df.count()
                
                removed = initial_count - final_count
                if removed > 0:
                    print(f"Đã loại bỏ {removed} outliers từ cột {col_name}")
    
    return df

def save_preprocessed_data(df, output_path):
    """Lưu dữ liệu đã preprocessing vào cả local và HDFS"""
    try:
        # Lưu vào local
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
        print(f"Đã lưu dữ liệu preprocessed tại local: {output_path}")
        
        # Thử lưu vào HDFS
        try:
            hdfs_output = "hdfs://localhost:9000/output/preprocessed_shipping_data"
            df.coalesce(1).write.mode("overwrite").option("header", "true").csv(hdfs_output)
            print(f"Đã lưu dữ liệu preprocessed tại HDFS: {hdfs_output}")
        except Exception as e:
            print(f"Không thể lưu vào HDFS: {e}")
        
        # In thống kê cuối cùng
        print(f"Tổng số records sau preprocessing: {df.count()}")
        print("Schema cuối cùng:")
        df.printSchema()
        
    except Exception as e:
        print(f"Lỗi khi lưu file: {e}")

def main():
    """Hàm chính chạy preprocessing pipeline"""
    # Tạo Spark session
    spark = create_spark_session()
    
    try:
        # Đường dẫn file input và output từ config
        input_path = FILE_PATHS["raw_data"]
        output_path = FILE_PATHS["preprocessed_data"]
        
        print("=== BẮT ĐẦU PREPROCESSING ===")
        
        # 1. Đọc dữ liệu thô
        print("1. Đọc dữ liệu thô...")
        df = load_raw_data(spark, input_path)
        if df is None:
            return
        
        # 2. Làm sạch tên cột
        print("2. Làm sạch tên cột...")
        df = clean_column_names(df)
        
        # 3. Xử lý missing values
        print("3. Xử lý missing values...")
        df = handle_missing_values(df)
        
        # 4. Làm sạch dữ liệu categorical
        print("4. Làm sạch dữ liệu categorical...")
        df = clean_categorical_data(df)
        
        # 5. Validate dữ liệu số
        print("5. Validate dữ liệu số...")
        df = validate_numerical_data(df)
        
        # 6. Tạo derived features
        print("6. Tạo derived features...")
        df = create_derived_features(df)
        
        # 7. Loại bỏ outliers
        print("7. Loại bỏ outliers...")
        df = remove_outliers(df)
        
        # 8. Lưu dữ liệu đã preprocessing
        print("8. Lưu dữ liệu...")
        save_preprocessed_data(df, output_path)
        
        print("=== HOÀN THÀNH PREPROCESSING ===")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
