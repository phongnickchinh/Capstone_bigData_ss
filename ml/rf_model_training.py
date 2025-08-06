import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Thêm các thư mục cần thiết vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spark_utils

def init_spark():
    """
    Khởi tạo SparkSession
    """
    spark = SparkSession.builder \
        .appName("Shipping Prediction - Random Forest") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()
    
    # Thiết lập log level
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_data(spark):
    """
    Đọc dữ liệu từ các tệp CSV
    """
    print("Đang đọc dữ liệu huấn luyện và kiểm tra...")
    
    # Đọc dữ liệu train từ đường dẫn tương đối
    train_paths = [
        "file:///mnt/p/coddd/Capstone_group4/ml/Train_ML.csv",
        "file:///mnt/p/coddd/Capstone_group4/Train_ML.csv"
    ]
    
    train_df = None
    for path in train_paths:
        try:
            train_df = spark.read.csv(path, header=True, inferSchema=True)
            print(f"Đã đọc dữ liệu train từ: {path}")
            break
        except Exception:
            continue
    
    if train_df is None:
        raise FileNotFoundError("Không tìm thấy file Train_ML.csv")
    
    # Đọc dữ liệu test
    test_paths = [
        "file:///mnt/p/coddd/Capstone_group4/ml/Test_ML.csv",
        "file:///mnt/p/coddd/Capstone_group4/Test_ML.csv"
    ]
    
    test_df = None
    for path in test_paths:
        try:
            test_df = spark.read.csv(path, header=True, inferSchema=True)
            print(f"Đã đọc dữ liệu test từ: {path}")
            break
        except Exception:
            continue
    
    if test_df is None:
        raise FileNotFoundError("Không tìm thấy file Test_ML.csv")
        
    # Loại bỏ cột ID nếu có
    if "ID" in train_df.columns:
        train_df = train_df.drop("ID")
    if "ID" in test_df.columns:
        test_df = test_df.drop("ID")

    #thêm df = df.withColumn("Loyalty_score", df["Customer_rating"] * df["Prior_purchases"])
    train_df = train_df.withColumn("Loyalty_score", col("Customer_rating") * col("Prior_purchases"))
    test_df = test_df.withColumn("Loyalty_score", col("Customer_rating") * col("Prior_purchases"))


    print(f"Đã đọc dữ liệu huấn luyện: {train_df.count()} dòng, {len(train_df.columns)} cột")
    print(f"Đã đọc dữ liệu kiểm tra: {test_df.count()} dòng, {len(test_df.columns)} cột")
    
    return train_df, test_df


def prepare_features(train_df, test_df):
    """
    Chuẩn bị features cho mô hình
    """
    print("Đang chuẩn bị features...")
    
    # Đổi tên cột để tránh lỗi với dấu chấm
    target_col = "Reached.on.Time_Y.N"
    target_col_new = "Reached_on_Time_Y_N"
    
    # Đảm bảo cột đích tồn tại trong dataset
    if target_col in train_df.columns:
        # Đổi tên cột trong cả train và test
        train_df = train_df.withColumnRenamed(target_col, target_col_new)
        test_df = test_df.withColumnRenamed(target_col, target_col_new)
    else:
        print(f"Cột {target_col} không tồn tại trong dataset. Có thể đã đổi tên trước đó.")
    
    # In ra các cột hiện có để kiểm tra
    print("Các cột hiện có trong dataset:", train_df.columns)
    
    # Xác định các loại cột
    categorical_cols = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]
    numeric_cols = [col for col in train_df.columns 
                   if col not in categorical_cols 
                   and col != target_col_new]
    
    #bỏ đi các cột phân loại do trước đó không có đóng góp cho mô hình
    # Chọn các cột số để tạo features
    
    # Chuyển đổi biến mục tiêu thành kiểu số (nếu cần)
    train_df = train_df.withColumn(target_col_new, col(target_col_new).cast("integer"))
    
    print("Các biến phân loại:", categorical_cols)
    print("Các biến số:", numeric_cols)
    
    # Tạo các bước xử lý trong pipeline
    # indexers = [StringIndexer(inputCol=col_name, outputCol=col_name+"_index", handleInvalid="keep") 
    #            for col_name in categorical_cols]
    
    # encoders = [OneHotEncoder(inputCol=col_name+"_index", outputCol=col_name+"_encoded")
    #            for col_name in categorical_cols]
    
    # # Chuẩn bị các cột cho Vector Assembler
    # encoded_cols = [col_name+"_encoded" for col_name in categorical_cols] #ignore
    feature_cols = numeric_cols
    
    # Tạo Vector Assembler
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep"
    )
    
    return categorical_cols, numeric_cols, assembler

def build_rf_pipeline( assembler):
    """
    Xây dựng pipeline cho Random Forest
    """
    print("Đang xây dựng pipeline cho Random Forest...")
    
    # Khởi tạo mô hình Random Forest
    rf = RandomForestClassifier(
        labelCol="Reached_on_Time_Y_N",
        featuresCol="features",
        numTrees=100,
        maxDepth=10,
        seed=42
    )
    
    # Xây dựng pipeline
    pipeline_stages = [assembler, rf]
    pipeline = Pipeline(stages=pipeline_stages)
    
    return pipeline, rf

def train_with_cross_validation(pipeline, rf, train_df):
    """
    Huấn luyện mô hình với Cross Validation để tìm siêu tham số tốt nhất
    """
    print("Đang huấn luyện mô hình với Cross Validation...")
    
    # Tạo grid tham số
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()
    
    # Tạo evaluator
    evaluator = BinaryClassificationEvaluator(
        labelCol="Reached_on_Time_Y_N",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    # Tạo Cross Validator
    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2,
        seed=42
    )
    
    # Huấn luyện mô hình
    print("Bắt đầu huấn luyện mô hình với Cross Validation...")
    cv_model = crossval.fit(train_df)
    print("Đã hoàn thành huấn luyện mô hình")
    
    # Lấy mô hình tốt nhất
    best_model = cv_model.bestModel
    
    # Lấy thông tin về tham số tốt nhất của Random Forest
    best_rf = best_model.stages[-1]
    print(f"Tham số tốt nhất: numTrees={best_rf.getNumTrees}, maxDepth={best_rf.getMaxDepth()}")
    
    return cv_model, best_model

def evaluate_model(model, train_df, test_df):
    """
    Đánh giá mô hình trên tập huấn luyện và kiểm tra
    """
    print("Đang đánh giá mô hình...")
    
    # Dự đoán trên tập huấn luyện và kiểm tra
    train_predictions = model.transform(train_df)
    test_predictions = model.transform(test_df)
    
    # Tạo evaluator cho các chỉ số khác nhau
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="Reached_on_Time_Y_N",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="Reached_on_Time_Y_N",
        predictionCol="prediction",
        metricName="accuracy"
    )
    
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="Reached_on_Time_Y_N",
        predictionCol="prediction",
        metricName="f1"
    )
    
    # Tính toán các chỉ số
    train_auc = evaluator_auc.evaluate(train_predictions)
    test_auc = evaluator_auc.evaluate(test_predictions)
    
    train_acc = evaluator_acc.evaluate(train_predictions)
    test_acc = evaluator_acc.evaluate(test_predictions)
    
    train_f1 = evaluator_f1.evaluate(train_predictions)
    test_f1 = evaluator_f1.evaluate(test_predictions)
    
    # Tính toán precision và recall
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="Reached_on_Time_Y_N",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="Reached_on_Time_Y_N",
        predictionCol="prediction",
        metricName="weightedRecall"
    )
    
    train_precision = evaluator_precision.evaluate(train_predictions)
    test_precision = evaluator_precision.evaluate(test_predictions)
    
    train_recall = evaluator_recall.evaluate(train_predictions)
    test_recall = evaluator_recall.evaluate(test_predictions)
    
    print("\n===== KẾT QUẢ ĐÁNH GIÁ =====")
    print(f"Tập huấn luyện - AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
    print(f"Tập kiểm tra - AUC: {test_auc:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    # Tạo dict để lưu tất cả các chỉ số
    evaluation_metrics = {
        'train': {
            'auc': train_auc,
            'accuracy': train_acc,
            'f1': train_f1,
            'precision': train_precision,
            'recall': train_recall
        },
        'test': {
            'auc': test_auc,
            'accuracy': test_acc,
            'f1': test_f1,
            'precision': test_precision,
            'recall': test_recall
        }
    }
    
    return train_predictions, test_predictions, evaluation_metrics

def extract_feature_importance(model):
    """
    Trích xuất và hiển thị độ quan trọng của các đặc trưng
    """
    print("\n===== ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG =====")
    
    # Lấy RF model từ pipeline
    rf_model = model.stages[-1]
    
    # Lấy Vector Assembler để biết thứ tự các đặc trưng
    assembler = model.stages[-2]
    feature_names = assembler.getInputCols()
    
    # Lấy độ quan trọng của các đặc trưng
    importances = rf_model.featureImportances.toArray()
    
    # Tạo danh sách đặc trưng và độ quan trọng
    feature_importance = [(feature, importance) for feature, importance in zip(feature_names, importances)]
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Tạo DataFrame và lưu vào CSV
    import pandas as pd
    df_importance = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs('ml/random_forest', exist_ok=True)

    # Lưu độ quan trọng vào file CSV
    df_importance.to_csv('ml/random_forest/rf_feature_importance.csv', index=False)
    # Lưu một bản sao ở thư mục gốc ml cho tương thích ngược
    df_importance.to_csv('ml/rf_feature_importance.csv', index=False)

    # In ra top 10 đặc trưng quan trọng nhất
    print("Top 10 đặc trưng quan trọng nhất:")
    for feature, importance in feature_importance[:10]:
        print(f"{feature}: {importance:.6f}")
    
    return feature_importance

def save_model(model, model_path):
    """
    Lưu mô hình đã huấn luyện
    """
    print(f"\nĐang lưu mô hình vào {model_path}...")
    model.write().overwrite().save(model_path)
    print(f"Đã lưu mô hình thành công!")

def main():
    """
    Hàm chính để thực hiện toàn bộ quá trình
    """
    # Khởi tạo Spark
    spark = init_spark()
    
    try:
        # Đọc dữ liệu
        train_df, test_df = load_data(spark)
        train_df = train_df.withColumnRenamed("Reached.on.Time_Y.N", "Reached_on_Time_Y_N")
        test_df = test_df.withColumnRenamed("Reached.on.Time_Y.N", "Reached_on_Time_Y_N")
        # Chuẩn bị features
        categorical_cols, numeric_cols, assembler = prepare_features(train_df, test_df)
        
        # Xây dựng pipeline
        pipeline, rf = build_rf_pipeline(assembler)

        # Huấn luyện mô hình với cross validation
        

        cv_model, best_model = train_with_cross_validation(pipeline, rf, train_df)
        
        # Đánh giá mô hình
        train_predictions, test_predictions, metrics = evaluate_model(best_model, train_df, test_df)
        
        # Trích xuất độ quan trọng của các đặc trưng
        feature_importance = extract_feature_importance(best_model)
        
        # Lưu mô hình
        model_path = "file:///mnt/p/coddd/Capstone_group4/ml/random_forest/rf_model"
        save_model(best_model, model_path)
        
        # Tạo file kết quả để so sánh các mô hình
        import pandas as pd
        results_df = pd.DataFrame({
            'Model': ['Random Forest'],
            'AUC': [metrics['test']['auc']],
            'Accuracy': [metrics['test']['accuracy']],
            'F1 Score': [metrics['test']['f1']],
            'Precision': [metrics['test']['precision']],
            'Recall': [metrics['test']['recall']]
        })
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs('ml/random_forest', exist_ok=True)
        
        # Lưu kết quả vào file CSV
        results_df.to_csv('ml/random_forest/rf_results.csv', index=False)
        # Lưu một bản sao ở thư mục gốc ml cho tương thích ngược
        results_df.to_csv('ml/rf_results.csv', index=False)

        print("\nQuá trình huấn luyện và đánh giá đã hoàn thành!")
        
    finally:
        # Dừng Spark session
        spark.stop()

if __name__ == "__main__":
    main()
