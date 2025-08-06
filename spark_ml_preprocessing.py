from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from analysis.config import SPARK_CONFIG, FILE_PATHS, COLUMN_MAPPINGS

def create_spark_session():
    """Tạo Spark session cho ML preprocessing"""
    builder = SparkSession.builder.appName("ML_Data_Preprocessing")
    
    # Thêm các config từ file config
    for key, value in SPARK_CONFIG["preprocessing"]["configs"].items():
        builder = builder.config(key, value)
    
    return builder.getOrCreate()

def load_cleaned_data(spark, file_path):
    """Đọc dữ liệu đã được cleaned từ preprocessing pipeline"""
    try:
        # Đọc từ preprocessed data thay vì raw data
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        print(f"Đã đọc {df.count()} records từ {file_path}")
        return df
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None

def prepare_ml_features(df):
    """Chuẩn bị features cho Machine Learning"""
    
    # 1. StringIndexer cho categorical features
    categorical_cols = ["warehouse_block", "mode_of_shipment", "product_importance"]
    if "gender" in df.columns:  # Nếu có cột gender
        categorical_cols.append("gender")
    
    indexers = []
    for col_name in categorical_cols:
        if col_name in df.columns:
            indexer = StringIndexer(
                inputCol=col_name, 
                outputCol=f"{col_name}_index",
                handleInvalid="keep"  # Handle unknown categories
            )
            indexers.append(indexer)
    
    # StringIndexer cho target variable
    if "reached_on_time" in df.columns:
        label_indexer = StringIndexer(
            inputCol="reached_on_time", 
            outputCol="label",
            handleInvalid="keep"
        )
        indexers.append(label_indexer)
    
    # 2. OneHotEncoder cho categorical features
    encoder_input_cols = [f"{col}_index" for col in categorical_cols if col in df.columns]
    encoder_output_cols = [f"{col}_ohe" for col in categorical_cols if col in df.columns]
    
    encoder = OneHotEncoder(
        inputCols=encoder_input_cols,
        outputCols=encoder_output_cols,
        handleInvalid="keep"
    )
    
    # 3. Numerical features để scaling
    numeric_cols = ["customer_care_calls", "customer_rating", "cost_of_product", 
                   "prior_purchases", "discount_offered", "weight_in_gms"]
    
    # Lọc chỉ những cột có trong DataFrame
    available_numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # VectorAssembler cho numeric features
    numeric_assembler = VectorAssembler(
        inputCols=available_numeric_cols, 
        outputCol="numeric_features",
        handleInvalid="keep"
    )
    
    # 4. StandardScaler cho numeric features
    scaler = StandardScaler(
        inputCol="numeric_features", 
        outputCol="scaled_features", 
        withMean=True, 
        withStd=True
    )
    
    # 5. Final VectorAssembler để gộp tất cả features
    final_feature_cols = encoder_output_cols + ["scaled_features"]
    
    final_assembler = VectorAssembler(
        inputCols=final_feature_cols,
        outputCol="features",
        handleInvalid="keep"
    )
    
    # 6. Tạo ML Pipeline
    stages = indexers + [encoder, numeric_assembler, scaler, final_assembler]
    
    pipeline = Pipeline(stages=stages)
    
    return pipeline

def create_ml_dataset(df, pipeline):
    """Tạo dataset cho machine learning"""
    # Fit và transform pipeline
    pipeline_model = pipeline.fit(df)
    transformed_df = pipeline_model.transform(df)
    
    # Select chỉ features và label cần thiết cho ML
    if "label" in transformed_df.columns:
        ml_df = transformed_df.select("features", "label")
    else:
        # Nếu không có label (unsupervised learning)
        ml_df = transformed_df.select("features")
    
    return ml_df, pipeline_model

def save_ml_dataset(ml_df, output_path):
    """Lưu ML dataset"""
    try:
        # Lưu dưới dạng parquet cho ML
        ml_df.write.mode("overwrite").parquet(output_path)
        print(f"Đã lưu ML dataset tại: {output_path}")
        
        # In thống kê
        print(f"Tổng số records trong ML dataset: {ml_df.count()}")
        print("Schema của ML dataset:")
        ml_df.printSchema()
        
        # Show sample
        print("Sample data:")
        ml_df.show(5, truncate=False)
        
    except Exception as e:
        print(f"Lỗi khi lưu ML dataset: {e}")

def save_pipeline_model(pipeline_model, model_path):
    """Lưu pipeline model để sử dụng sau"""
    try:
        pipeline_model.write().overwrite().save(model_path)
        print(f"Đã lưu pipeline model tại: {model_path}")
    except Exception as e:
        print(f"Lỗi khi lưu pipeline model: {e}")

def calculate_correlation_matrix(df, pipeline_model, save_path=None):
    """Tính ma trận tương quan từ dữ liệu đã được transform"""
    try:
        print("6. Tính ma trận tương quan...")
        
        # Transform dữ liệu bằng pipeline đã fit
        transformed_df = pipeline_model.transform(df)
        
        # Lấy các cột numeric để tính correlation
        numeric_cols = ["customer_care_calls", "customer_rating", "cost_of_product", 
                       "prior_purchases", "discount_offered", "weight_in_gms"]
        
        # Thêm các cột index từ categorical features
        categorical_index_cols = []
        categorical_cols = ["warehouse_block", "mode_of_shipment", "product_importance"]
        if "gender" in df.columns:
            categorical_cols.append("gender")
        
        for col_name in categorical_cols:
            if col_name in df.columns:
                categorical_index_cols.append(f"{col_name}_index")
        
        # Thêm target variable nếu có
        target_cols = []
        if "reached_on_time" in df.columns:
            # Tạo numeric version của target
            transformed_df = transformed_df.withColumn("reached_on_time_numeric", 
                                                     col("reached_on_time").cast("double"))
            target_cols.append("reached_on_time_numeric")
        
        # Gộp tất cả cột để tính correlation
        correlation_cols = [col for col in numeric_cols if col in df.columns] + \
                          categorical_index_cols + target_cols
        
        print(f"Các cột để tính correlation: {correlation_cols}")
        
        # Chuyển sang Pandas để tính correlation (vì PySpark không có correlation matrix built-in)
        if len(correlation_cols) > 1:
            # Select và filter null values
            corr_df = transformed_df.select(correlation_cols).na.drop()
            
            # Convert to Pandas
            corr_pandas = corr_df.toPandas()
            
            # Tính correlation matrix
            correlation_matrix = corr_pandas.corr()
            
            print("Ma trận tương quan:")
            print(correlation_matrix.round(3))
            
            # Visualize correlation matrix
            visualize_correlation_matrix(correlation_matrix, save_path)
            
            # Lưu correlation matrix thành CSV
            if save_path:
                csv_path = f"{save_path}_correlation_matrix.csv"
                correlation_matrix.to_csv(csv_path)
                print(f"Đã lưu correlation matrix tại: {csv_path}")
            
            return correlation_matrix
        else:
            print("Không đủ cột numeric để tính correlation matrix")
            return None
            
    except Exception as e:
        print(f"Lỗi khi tính correlation matrix: {e}")
        return None

def visualize_correlation_matrix(correlation_matrix, save_path=None):
    """Vẽ heatmap cho correlation matrix"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Tạo figure với kích thước phù hợp
        plt.figure(figsize=(12, 10))
        
        # Tạo heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(correlation_matrix, 
                    annot=True, 
                    cmap='coolwarm', 
                    center=0,
                    fmt='.2f',
                    mask=mask,
                    square=True,
                    cbar_kws={"shrink": .8})
        
        plt.title('Ma Trận Tương Quan (Correlation Matrix)', fontsize=16, pad=20)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Lưu figure nếu có đường dẫn
        if save_path:
            plot_path = f"{save_path}_correlation_plot.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu correlation plot tại: {plot_path}")
        
        # Hiển thị plot
        plt.show()
        
        # In các correlation cao nhất
        print("\n🔍 TOP CORRELATIONS:")
        print("=" * 50)
        
        # Flatten correlation matrix và sort
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if not pd.isna(corr_val):
                    corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j], 
                        'correlation': corr_val
                    })
        
        # Sort theo absolute correlation
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # In top 10 correlations
        print("Top 10 highest correlations:")
        for i, pair in enumerate(corr_pairs[:10]):
            print(f"{i+1:2d}. {pair['feature1']:<25} vs {pair['feature2']:<25}: {pair['correlation']:+.3f}")
            
    except Exception as e:
        print(f"Lỗi khi vẽ correlation matrix: {e}")

def analyze_feature_importance_from_correlation(correlation_matrix, target_col="reached_on_time_numeric"):
    """Phân tích tầm quan trọng của features dựa trên correlation với target"""
    try:
        if target_col in correlation_matrix.columns:
            # Lấy correlation với target variable
            target_correlations = correlation_matrix[target_col].abs().sort_values(ascending=False)
            
            print(f"\n📊 FEATURE IMPORTANCE (based on correlation with {target_col}):")
            print("=" * 70)
            
            for i, (feature, corr) in enumerate(target_correlations.items()):
                if feature != target_col and not pd.isna(corr):
                    importance_level = "🔴 High" if corr > 0.5 else "🟡 Medium" if corr > 0.3 else "🟢 Low"
                    print(f"{i+1:2d}. {feature:<30}: {corr:.3f} {importance_level}")
            
            return target_correlations
        else:
            print(f"Target column '{target_col}' not found in correlation matrix")
            return None
            
    except Exception as e:
        print(f"Lỗi khi phân tích feature importance: {e}")
        return None

def main():
    """Hàm chính cho ML preprocessing"""
    spark = create_spark_session()
    
    try:
        # Đọc dữ liệu đã được cleaned
        cleaned_data_path = FILE_PATHS["preprocessed_data"] + "/*.csv"
        print("=== BẮT ĐẦU ML PREPROCESSING ===")
        
        # 1. Load cleaned data
        print("1. Đọc dữ liệu đã cleaned...")
        df = load_cleaned_data(spark, cleaned_data_path)
        if df is None:
            print("Không thể đọc dữ liệu. Hãy chạy spark_preprocessing.py trước.")
            return
        
        # 2. Chuẩn bị ML pipeline
        print("2. Tạo ML preprocessing pipeline...")
        pipeline = prepare_ml_features(df)
        
        # 3. Transform data
        print("3. Transform dữ liệu cho ML...")
        ml_df, pipeline_model = create_ml_dataset(df, pipeline)
        
        # 4. Lưu ML dataset
        print("4. Lưu ML dataset...")
        ml_output_path = "ml_dataset"
        save_ml_dataset(ml_df, ml_output_path)
        
        # 5. Lưu pipeline model
        print("5. Lưu pipeline model...")
        model_path = "ml_pipeline_model"
        save_pipeline_model(pipeline_model, model_path)
        
        # 6. Tính correlation matrix
        correlation_matrix = calculate_correlation_matrix(df, pipeline_model, "correlation_analysis")
        
        # 7. Phân tích feature importance
        if correlation_matrix is not None:
            feature_importance = analyze_feature_importance_from_correlation(correlation_matrix)
        
        print("=== HOÀN THÀNH ML PREPROCESSING ===")
        
        return ml_df, pipeline_model, correlation_matrix
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
