import os
import sys
import time

def run_pipeline():
    """
    Chạy toàn bộ quy trình huấn luyện và đánh giá mô hình CatBoost
    """
    print("="*80)
    print("KHỞI CHẠY QUY TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH CATBOOST")
    print("="*80)

    # Đường dẫn tới các file script
    catboost_script = "ml/catboost_model_training.py"
    compare_script = "ml/compare_models.py"

    # Kiểm tra và tạo thư mục ml nếu chưa tồn tại
    if not os.path.exists("ml"):
        os.makedirs("ml")
        print("Đã tạo thư mục ml")

    # Bước 1: Cài đặt các thư viện cần thiết
    print("\n[Bước 1] Đang cài đặt các thư viện cần thiết...")
    start_time = time.time()
    try:
        os.system("pip install catboost pandas scikit-learn matplotlib seaborn numpy")
        print(f"\n[Bước 1] Đã hoàn thành cài đặt sau {time.time() - start_time:.2f} giây")
    except Exception as e:
        print(f"Lỗi trong quá trình cài đặt thư viện: {str(e)}")
        return

    # Bước 2: Huấn luyện mô hình CatBoost
    print("\n[Bước 2] Đang huấn luyện mô hình CatBoost...")
    start_time = time.time()
    try:
        os.system(f"python {catboost_script}")
        print(f"\n[Bước 2] Đã hoàn thành huấn luyện mô hình sau {time.time() - start_time:.2f} giây")
    except Exception as e:
        print(f"Lỗi trong quá trình huấn luyện mô hình: {str(e)}")
        return

    # Bước 3: So sánh các mô hình
    print("\n[Bước 3] Đang so sánh các mô hình...")
    start_time = time.time()
    try:
        os.system(f"python {compare_script}")
        print(f"\n[Bước 3] Đã hoàn thành so sánh mô hình sau {time.time() - start_time:.2f} giây")
    except Exception as e:
        print(f"Lỗi trong quá trình so sánh mô hình: {str(e)}")
        return

    print("\n="*80)
    print("QUY TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH CATBOOST ĐÃ HOÀN THÀNH")
    print("="*80)
    print("\nCác file kết quả được lưu trong thư mục ml:")
    print("- Mô hình: ml/catboost/catboost_model.cbm")
    print("- Confusion Matrix: ml/catboost/catboost_confusion_matrix.png")
    print("- Độ quan trọng của các đặc trưng: ml/catboost/catboost_feature_importance.png")
    print("- Đường cong ROC: ml/catboost/catboost_roc_curve.png")
    print("- Các chỉ số hiệu suất: ml/catboost/catboost_performance_metrics.png")
    print("- So sánh giữa các mô hình: ml/comparisons/model_comparison_chart.png")
    print("- So sánh độ quan trọng của các đặc trưng: ml/comparisons/feature_importance_comparison.png")

if __name__ == "__main__":
    run_pipeline()
