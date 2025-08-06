# Hướng dẫn sử dụng mô hình Random Forest và CatBoost

## Cài đặt các phụ thuộc

Trước khi chạy các script, hãy đảm bảo bạn đã cài đặt tất cả các phụ thuộc cần thiết:

```bash
pip install -r ml/requirements.txt
```

## Cài đặt PySpark

Đảm bảo bạn đã cài đặt Apache Spark và thiết lập các biến môi trường cần thiết:

- Cài đặt Java JDK (nếu chưa có)
- Tải và giải nén Apache Spark
- Thiết lập biến môi trường:
  - `SPARK_HOME`: đường dẫn tới thư mục Spark
  - `JAVA_HOME`: đường dẫn tới Java JDK
  - Thêm `%SPARK_HOME%\bin` vào biến PATH

## Cấu trúc dự án

```
ml/
  ├── rf_model_training.py          # Script huấn luyện mô hình Random Forest
  ├── rf_model_evaluation.py        # Script đánh giá mô hình Random Forest
  ├── catboost_model_training.py    # Script huấn luyện và đánh giá mô hình CatBoost
  ├── compare_feature_importance.py # So sánh độ quan trọng của các đặc trưng (RF)
  ├── compare_models.py             # So sánh các mô hình (RF và CatBoost)
  ├── run_rf_pipeline.py            # Script chạy toàn bộ quy trình Random Forest
  ├── run_catboost_pipeline.py      # Script chạy toàn bộ quy trình CatBoost
  ├── Train_ML.csv                  # Dữ liệu huấn luyện đã xử lý
  └── Test_ML.csv                   # Dữ liệu kiểm tra đã xử lý
```

## Sử dụng

### Chạy toàn bộ quy trình

#### Mô hình Random Forest

Để chạy toàn bộ quy trình huấn luyện và đánh giá mô hình Random Forest:

```bash
python ml/run_rf_pipeline.py
```

#### Mô hình CatBoost

Để chạy toàn bộ quy trình huấn luyện và đánh giá mô hình CatBoost:

```bash
python ml/run_catboost_pipeline.py
```

#### So sánh các mô hình

Để so sánh hiệu suất giữa các mô hình:

```bash
python ml/compare_models.py
```

### Chạy từng bước riêng biệt

1. Huấn luyện mô hình Random Forest:

```bash
python ml/rf_model_training.py
```

2. Đánh giá mô hình Random Forest:

```bash
python ml/rf_model_evaluation.py
```

3. Huấn luyện và đánh giá mô hình CatBoost:

```bash
python ml/catboost_model_training.py
```

4. So sánh độ quan trọng của các đặc trưng (RF):

```bash
python ml/compare_feature_importance.py
```

## Kết quả

Sau khi chạy các script, kết quả sẽ được lưu trong thư mục `ml/`:

### Mô hình Random Forest
- Mô hình: `ml/rf_model/`
- Confusion Matrix: `ml/rf_confusion_matrix.png`
- Độ quan trọng của các đặc trưng: `ml/rf_feature_importance.png`
- Đường cong ROC: `ml/rf_roc_curve.png`
- Các chỉ số hiệu suất: `ml/rf_performance_metrics.png`
- So sánh độ quan trọng: `ml/feature_importance_comparison.png`

### Mô hình CatBoost
- Mô hình: `ml/catboost_model.cbm`
- Confusion Matrix: `ml/catboost_confusion_matrix.png`
- Độ quan trọng của các đặc trưng: `ml/catboost_feature_importance.png`
- Đường cong ROC: `ml/catboost_roc_curve.png`
- Các chỉ số hiệu suất: `ml/catboost_performance_metrics.png`

### So sánh mô hình
- So sánh hiệu suất: `ml/model_comparison_chart.png`
- So sánh độ quan trọng các đặc trưng: `ml/feature_importance_comparison_models.png`

## Các thông số mô hình

### Mô hình Random Forest
- Số cây (numTrees): 50, 100 (chọn tốt nhất qua Cross Validation)
- Độ sâu tối đa (maxDepth): 5, 10 (chọn tốt nhất qua Cross Validation)
- Số fold cho Cross Validation: 3

### Mô hình CatBoost
- Số vòng lặp (iterations): 1000 với early stopping
- Tốc độ học (learning_rate): 0.03
- Độ sâu cây (depth): 6
- Độ chính quy L2 (l2_leaf_reg): 3
- Hàm mất mát (loss_function): Logloss
- Chỉ số đánh giá (eval_metric): AUC

## Xử lý dữ liệu

### Mô hình Random Forest (PySpark)

Quá trình xử lý dữ liệu với PySpark bao gồm:

- Chuyển đổi các biến phân loại thành số bằng StringIndexer và OneHotEncoder
- Kết hợp các đặc trưng thành một vector bằng VectorAssembler
- Huấn luyện mô hình Random Forest trên dữ liệu đã xử lý
- Đánh giá mô hình bằng các chỉ số như AUC, Accuracy, F1, Precision, Recall

### Mô hình CatBoost

CatBoost có khả năng xử lý trực tiếp các biến phân loại, quá trình xử lý dữ liệu bao gồm:

- Xác định các biến phân loại và chuyển thành kiểu dữ liệu category
- Sử dụng Pool để cung cấp dữ liệu cho mô hình với các biến phân loại được xác định
- CatBoost tự động xử lý các biến phân loại mà không cần one-hot encoding
- Huấn luyện mô hình với early stopping để tránh overfitting
- Đánh giá mô hình bằng các chỉ số như AUC, Accuracy, F1, Precision, Recall

## Chú ý

- Đảm bảo rằng dữ liệu Train_ML.csv và Test_ML.csv đã được chuẩn bị trước khi chạy các script
- Nếu không tìm thấy các file này trong thư mục `ml/`, script sẽ tìm kiếm trong thư mục gốc
