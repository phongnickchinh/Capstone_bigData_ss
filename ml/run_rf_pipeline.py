import os
import sys
import time

def run_pipeline():
    """
    Chạy toàn bộ quy trình huấn luyện và đánh giá mô hình Random Forest
    """
    print("="*80)
    print("KHỞI CHẠY QUY TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST")
    print("="*80)

    # Đường dẫn tới các file script
    training_script = "ml/rf_model_training.py"
    evaluation_script = "ml/rf_model_evaluation.py"

    # Kiểm tra và tạo thư mục ml nếu chưa tồn tại
    if not os.path.exists("ml"):
        os.makedirs("ml")
        print("Đã tạo thư mục ml")

    # Bước 1: Huấn luyện mô hình
    print("\n[Bước 1] Đang huấn luyện mô hình Random Forest...")
    start_time = time.time()
    try:
        os.system(f"python {training_script}")
        print(f"\n[Bước 1] Đã hoàn thành huấn luyện mô hình sau {time.time() - start_time:.2f} giây")
    except Exception as e:
        print(f"Lỗi trong quá trình huấn luyện mô hình: {str(e)}")
        return

    # Bước 2: Đánh giá mô hình
    print("\n[Bước 2] Đang đánh giá mô hình...")
    start_time = time.time()
    try:
        os.system(f"python {evaluation_script}")
        print(f"\n[Bước 2] Đã hoàn thành đánh giá mô hình sau {time.time() - start_time:.2f} giây")
    except Exception as e:
        print(f"Lỗi trong quá trình đánh giá mô hình: {str(e)}")
        return

    print("\n="*80)
    print("QUY TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH ĐÃ HOÀN THÀNH")
    print("="*20)
    print("\nCác file kết quả được lưu trong thư mục ml:")
    print("- Mô hình: ml/random_forest/rf_model")
    print("- Confusion Matrix: ml/random_forest/rf_confusion_matrix.png")
    print("- Độ quan trọng của các đặc trưng: ml/random_forest/rf_feature_importance.png")
    print("- Đường cong ROC: ml/random_forest/rf_roc_curve.png")
    print("- Các chỉ số hiệu suất: ml/random_forest/rf_performance_metrics.png")

if __name__ == "__main__":
    run_pipeline()
