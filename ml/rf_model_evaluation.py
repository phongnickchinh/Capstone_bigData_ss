import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col

# Thêm các thư mục cần thiết vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spark_utils

def init_spark():
    """
    Khởi tạo SparkSession
    """
    import os
    os.environ["PYSPARK_PYTHON"] = "/mnt/p/coddd/Capstone_group4/.venv310/bin/python3"  # Hoặc đúng path python bạn đang dùng
    os.environ["PYSPARK_DRIVER_PYTHON"] = "/mnt/p/coddd/Capstone_group4/.venv310/bin/python3"

    spark = SparkSession.builder \
        .appName("Shipping Prediction - Model Evaluation") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()
    
    # Thiết lập log level
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def load_data_and_model(spark, model_path):
    """
    Đọc dữ liệu và mô hình đã huấn luyện
    """
    print("Đang đọc dữ liệu và mô hình...")
    
    # Đọc dữ liệu test từ nhiều đường dẫn có thể
    test_paths = [
        "file:///mnt/p/coddd/Capstone_group4/ml/Test_ML.csv",
        "file:///mnt/p/coddd/Capstone_group4/Test_ML.csv",
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
    if "ID" in test_df.columns:
        test_df = test_df.drop("ID")
    
    # Đổi tên cột để tránh lỗi với dấu chấm
    target_col = "Reached.on.Time_Y.N"
    target_col_new = "Reached_on_Time_Y_N"
    
    # Đổi tên cột
    test_df = test_df.withColumnRenamed(target_col, target_col_new)
    
    # Chuyển đổi biến mục tiêu thành kiểu số
    test_df = test_df.withColumn(target_col_new, col(target_col_new).cast("integer"))

    #thêm loyalty_score
    test_df = test_df.withColumn("Loyalty_score", col("Customer_rating") * col("Prior_purchases"))

    # Đọc mô hình
    model = PipelineModel.load(model_path)
    
    print(f"Đã đọc dữ liệu kiểm tra: {test_df.count()} dòng, {len(test_df.columns)} cột")
    print(f"Đã đọc mô hình từ {model_path}")
    
    return test_df, model

def evaluate_model_detailed(model, test_df):
    """
    Đánh giá chi tiết mô hình
    """
    print("\nĐang đánh giá mô hình chi tiết...")
    
    # Dự đoán trên tập kiểm tra
    predictions = model.transform(test_df)
    
    # Tạo evaluator cho các chỉ số khác nhau
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="Reached_on_Time_Y_N",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="Reached_on_Time_Y_N",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
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
    
    # Tính toán các chỉ số
    auc = evaluator_auc.evaluate(predictions)
    pr_auc = evaluator_pr.evaluate(predictions)
    accuracy = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    
    # In kết quả
    print("\n===== CHI TIẾT KẾT QUẢ ĐÁNH GIÁ =====")
    print(f"AUC: {auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Tính confusion matrix
    pred_and_labels = predictions.select("prediction", "Reached_on_Time_Y_N")
    pred_and_labels = pred_and_labels.withColumn("prediction", col("prediction").cast("double"))
    pred_and_labels = pred_and_labels.withColumn("Reached_on_Time_Y_N", col("Reached_on_Time_Y_N").cast("double"))
    
    metrics = MulticlassMetrics(pred_and_labels.rdd.map(lambda x: (float(x[0]), float(x[1]))))
    confusion_matrix = metrics.confusionMatrix().toArray()
    
    print("\n===== CONFUSION MATRIX =====")
    print("Predicted / Actual  |  0 (Không đúng hạn)  |  1 (Đúng hạn)")
    print(f"0 (Không đúng hạn)  |  {int(confusion_matrix[0][0])}  |  {int(confusion_matrix[0][1])}")
    print(f"1 (Đúng hạn)        |  {int(confusion_matrix[1][0])}  |  {int(confusion_matrix[1][1])}")
    
    # Chuyển đổi dự đoán sang pandas để vẽ biểu đồ
    pandas_predictions = predictions.select("Reached_on_Time_Y_N", "prediction", "probability").toPandas()
    
    return pandas_predictions, auc, accuracy, f1, precision, recall, confusion_matrix

def plot_confusion_matrix(confusion_matrix):
    """
    Vẽ confusion matrix
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ["Không đúng hạn", "Đúng hạn"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Hiển thị giá trị trong ô
    thresh = confusion_matrix.max() / 2
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(int(confusion_matrix[i, j]), 'd'),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.ylabel('Giá trị dự đoán')
    plt.xlabel('Giá trị thực tế')
    plt.tight_layout()
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs('ml/random_forest', exist_ok=True)
    plt.savefig('ml/random_forest/rf_confusion_matrix.png', dpi=300)
    plt.close()

def extract_feature_importance(model):
    """
    Trích xuất độ quan trọng của các đặc trưng và vẽ biểu đồ
    """
    print("\nĐang trích xuất độ quan trọng của các đặc trưng...")

    # Lấy RF model từ pipeline
    rf_model = model.stages[-1]

    # Lấy Vector Assembler để biết thứ tự các đặc trưng
    assembler = model.stages[-2]
    feature_names = assembler.getInputCols()

    # Lấy độ quan trọng
    importances = rf_model.featureImportances.toArray()

    print(f"[DEBUG] Số features ban đầu: {len(feature_names)}")
    print(f"[DEBUG] Số importances: {len(importances)}")

    # Cắt về cùng độ dài nếu lệch
    if len(feature_names) != len(importances):
        min_len = min(len(feature_names), len(importances))
        print(f"[WARN] Độ dài không khớp! Cắt về {min_len}")
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]

    # Tạo DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance_df['Feature'][:15], feature_importance_df['Importance'][:15], color='skyblue')
    plt.xlabel('Độ quan trọng')
    plt.ylabel('Đặc trưng')
    plt.title('Top 15 đặc trưng quan trọng nhất - Random Forest')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('ml/random_forest/rf_feature_importance.png', dpi=300)
    plt.close()

    print("\n===== TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT =====")
    for i, (feature, importance) in enumerate(zip(feature_importance_df['Feature'][:15], feature_importance_df['Importance'][:15]), 1):
        print(f"{i}. {feature}: {importance:.6f}")

    return feature_importance_df


def plot_roc_curve(pandas_predictions, auc):
    """
    Vẽ đường cong ROC
    """
    from sklearn.metrics import roc_curve, auc as sk_auc
    
    # Extract probability for class 1
    pandas_predictions['probability_class1'] = pandas_predictions['probability'].apply(lambda x: float(x[1]))
    
    fpr, tpr, thresholds = roc_curve(pandas_predictions['Reached_on_Time_Y_N'], pandas_predictions['probability_class1'])
    roc_auc = sk_auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig('ml/random_forest/rf_roc_curve.png', dpi=300)
    plt.close()

def plot_performance_metrics(auc, accuracy, f1, precision, recall):
    """
    Vẽ biểu đồ hiển thị các chỉ số hiệu suất
    """
    metrics = ['AUC', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
    values = [auc, accuracy, f1, precision, recall]
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.ylim([0, 1.1])
    plt.ylabel('Score')
    plt.title('Chỉ số hiệu suất của mô hình Random Forest')
    
    # Thêm giá trị lên đỉnh thanh
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('ml/random_forest/rf_performance_metrics.png', dpi=300)
    plt.close()

def main():
    """
    Hàm chính để thực hiện đánh giá mô hình
    """
    # Khởi tạo Spark
    spark = init_spark()
    
    try:
        # Đường dẫn đến mô hình đã huấn luyện - thử nhiều đường dẫn khác nhau
        model_paths = [
            "file:///mnt/p/coddd/Capstone_group4/ml/random_forest/rf_model",
            "file:///mnt/p/coddd/Capstone_group4/ml/rf_model"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Tìm thấy mô hình tại: {path}")
                break
        
        if model_path is None:
            model_path = "file:///mnt/p/coddd/Capstone_group4/ml/random_forest/rf_model"
            print(f"Không tìm thấy mô hình, sẽ thử với đường dẫn mặc định: {model_path}")
        
        # Đọc dữ liệu và mô hình
        test_df, model = load_data_and_model(spark, model_path)
        
        # Đánh giá chi tiết mô hình
        pandas_predictions, auc, accuracy, f1, precision, recall, confusion_matrix = evaluate_model_detailed(model, test_df)
        
        # Vẽ confusion matrix
        plot_confusion_matrix(confusion_matrix)
        print("Đã lưu confusion matrix vào ml/random_forest/rf_confusion_matrix.png")
        
        # Trích xuất và vẽ độ quan trọng của các đặc trưng
        feature_importance_df = extract_feature_importance(model)
        print("Đã lưu biểu đồ độ quan trọng vào ml/random_forest/rf_feature_importance.png")
        
        # Vẽ đường cong ROC
        plot_roc_curve(pandas_predictions, auc)
        print("Đã lưu đường cong ROC vào ml/random_forest/rf_roc_curve.png")
        
        # Vẽ biểu đồ hiển thị các chỉ số hiệu suất
        plot_performance_metrics(auc, accuracy, f1, precision, recall)
        print("Đã lưu biểu đồ chỉ số hiệu suất vào ml/random_forest/rf_performance_metrics.png")
        
        print("\nĐã hoàn thành đánh giá mô hình!")
        
    finally:
        # Dừng Spark session
        spark.stop()

if __name__ == "__main__":
    main()
