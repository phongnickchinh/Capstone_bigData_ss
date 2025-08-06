"""
Configuration file cho Spark Processing Pipeline
"""

# Database Configuration

DATABASE_CONFIG = {
    "mysql_url": "jdbc:mysql://root:BOyeexNMytoXbEtZTkiZLhWCNbSUXrpj@hopper.proxy.rlwy.net:53175/railway",
    "mysql_props": {
        "user": "root",
        "password": "BOyeexNMytoXbEtZTkiZLhWCNbSUXrpj",
        "driver": "com.mysql.cj.jdbc.Driver",
        "connectTimeout": "30000",
        "socketTimeout": "30000"
    }
}

# File Paths
FILE_PATHS = {
    "raw_data": "Train_cleaned.csv",
    "preprocessed_data": "preprocessed_shipping_data",
    "ml_dataset": "outputs/models/ml_dataset",
    "ml_pipeline_model": "outputs/models/ml_pipeline_model", 
    "correlation_output": "outputs/correlation",
    "analysis_output": "outputs/analysis",
    "hdfs_raw": "hdfs://localhost:9000/input/Train_cleaned.csv",
    "hdfs_preprocessed": "hdfs://localhost:9000/output/preprocessed_shipping_data/*.csv"
}

# Spark Configuration
SPARK_CONFIG = {
    "preprocessing": {
        "app_name": "ShippingDataPreprocessing",
        "configs": {
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true"
        }
    },
    "processing": {
        "app_name": "ShippingPerformanceWithMySQL",
        "configs": {
            "spark.jars": "/home/phamp/jars/mysql-connector-j-8.3.0.jar",
            "spark.sql.warehouse.dir": "/tmp/spark-warehouse"
        }
    },
    "ml_preprocessing": {
        "app_name": "ML_Data_Preprocessing",
        "configs": {
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer"
        }
    }
}

# Data Quality Thresholds
DATA_QUALITY = {
    "max_missing_percentage": 30,  # % missing values tối đa cho phép
    "outlier_method": "IQR",       # Phương pháp detect outliers
    "outlier_multiplier": 1.5,     # Multiplier cho IQR method
    "min_records": 100             # Số records tối thiểu cần thiết
}

# Column Mappings (Raw -> Preprocessed)
COLUMN_MAPPINGS = {
    "ID": "id",
    "Warehouse_block": "warehouse_block",
    "Mode_of_Shipment": "mode_of_shipment",
    "Customer_care_calls": "customer_care_calls", 
    "Customer_rating": "customer_rating",
    "Cost_of_the_Product": "cost_of_product",
    "Prior_purchases": "prior_purchases",
    "Product_importance": "product_importance",
    "Gender": "gender",
    "Discount_offered": "discount_offered",
    "Weight_in_gms": "weight_in_gms",
    "Reached.on.Time_Y.N": "reached_on_time"
}

# Analysis Tables
ANALYSIS_TABLES = {
    "overall_performance": "analysis_01_overall_performance",
    "warehouse_analysis": "analysis_02_warehouse_block",
    "shipping_mode_analysis": "analysis_03_shipping_mode", 
    "product_importance_analysis": "analysis_04_product_importance",
    "discount_level_analysis": "analysis_05_discount_level",
    "weight_category_analysis": "analysis_06_weight_category",
    "combined_mode_weight": "analysis_07_combined_mode_weight",
    "customer_rating_analysis": "analysis_08_customer_rating",
    "customer_care_analysis": "analysis_09_customer_care_calls",
    "prior_purchases_analysis": "analysis_10_prior_purchases"
}

# Data Validation Rules
VALIDATION_RULES = {
    "customer_rating": {"min": 1, "max": 5, "default": 3},
    "discount_offered": {"min": 0, "max": 100, "default": 0},
    "weight_in_gms": {"min": 1, "default": 1000},
    "cost_of_product": {"min": 1, "default": 100},
    "customer_care_calls": {"min": 0, "max": 20, "default": 2}
}

# Feature Engineering Rules
FEATURE_ENGINEERING = {
    "weight_categories": {
        "Light": {"max": 1000},
        "Medium": {"min": 1001, "max": 5000},
        "Heavy": {"min": 5001}
    },
    "cost_categories": {
        "Low": {"max": 100},
        "Medium": {"min": 101, "max": 300},
        "High": {"min": 301}
    },
    "high_maintenance_threshold": {
        "min_calls": 4,
        "importance_level": "high"
    }
}
