import os
import sys
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc as sk_auc
import matplotlib.pyplot as plt
import seaborn as sns

# Thêm các thư mục cần thiết vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data():
    """
    Đọc dữ liệu huấn luyện và kiểm tra từ các tệp CSV
    """
    print("Đang đọc dữ liệu huấn luyện và kiểm tra...")
    
    # Đọc dữ liệu train
    train_path = "ml/Train_ML.csv"
    if not os.path.exists(train_path):
        train_path = "Train_ML.csv"
    
    train_df = pd.read_csv(train_path)
    
    # Đọc dữ liệu test
    test_path = "ml/Test_ML.csv"
    if not os.path.exists(test_path):
        test_path = "Test_ML.csv"
    
    test_df = pd.read_csv(test_path)
    
    # Loại bỏ cột ID nếu có
    if "ID" in train_df.columns:
        train_df = train_df.drop("ID", axis=1)
    if "ID" in test_df.columns:
        test_df = test_df.drop("ID", axis=1)
    
    print(f"Đã đọc dữ liệu huấn luyện: {train_df.shape[0]} dòng, {train_df.shape[1]} cột")
    print(f"Đã đọc dữ liệu kiểm tra: {test_df.shape[0]} dòng, {test_df.shape[1]} cột")
    
    return train_df, test_df

def prepare_features(train_df, test_df):
    """
    Chuẩn bị features cho mô hình
    """
    print("Đang chuẩn bị features...")
    
    # Xác định các loại cột
    categorical_cols = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]
    
    # Kiểm tra và chuyển đổi kiểu dữ liệu nếu cần
    for df in [train_df, test_df]:
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
    
    # Tách biến mục tiêu và đặc trưng
    X_train = train_df.drop("Reached.on.Time_Y.N", axis=1)
    y_train = train_df["Reached.on.Time_Y.N"]
    
    X_test = test_df.drop("Reached.on.Time_Y.N", axis=1)
    y_test = test_df["Reached.on.Time_Y.N"]
    
    print("Các biến phân loại:", categorical_cols)
    print("Các biến số:", [col for col in X_train.columns if col not in categorical_cols])
    
    return X_train, y_train, X_test, y_test, categorical_cols

def train_catboost_model(X_train, y_train, X_test, y_test, categorical_cols):
    """
    Huấn luyện mô hình CatBoost
    """
    print("Đang huấn luyện mô hình CatBoost...")
    
    # Tạo CatBoost Pool với chỉ định các biến phân loại
    train_pool = Pool(data=X_train, 
                     label=y_train,
                     cat_features=categorical_cols)
    
    test_pool = Pool(data=X_test,
                    label=y_test,
                    cat_features=categorical_cols)
    
    # Khởi tạo mô hình CatBoost
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100
    )
    
    # Huấn luyện mô hình với early stopping
    model.fit(
        train_pool,
        eval_set=test_pool,
        early_stopping_rounds=50,
        verbose=100
    )
    
    print(f"Đã huấn luyện mô hình trong {model.tree_count_} vòng lặp")
    
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test, categorical_cols):
    """
    Đánh giá mô hình trên tập huấn luyện và kiểm tra
    """
    print("Đang đánh giá mô hình...")
    
    # Dự đoán xác suất và nhãn
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Tính toán các chỉ số trên tập huấn luyện
    train_accuracy = accuracy_score(y_train, train_preds)
    train_f1 = f1_score(y_train, train_preds)
    train_precision = precision_score(y_train, train_preds)
    train_recall = recall_score(y_train, train_preds)
    train_auc = roc_auc_score(y_train, train_probs)
    
    # Tính toán các chỉ số trên tập kiểm tra
    test_accuracy = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds)
    test_recall = recall_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    
    # In kết quả
    print("\n===== KẾT QUẢ ĐÁNH GIÁ =====")
    print(f"Tập huấn luyện - AUC: {train_auc:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
    print(f"Tập kiểm tra - AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
    
    # Tính toán confusion matrix
    test_cm = confusion_matrix(y_test, test_preds)
    
    print("\n===== CONFUSION MATRIX (TEST) =====")
    print("Predicted / Actual  |  0 (Không đúng hạn)  |  1 (Đúng hạn)")
    print(f"0 (Không đúng hạn)  |  {test_cm[0][0]}  |  {test_cm[0][1]}")
    print(f"1 (Đúng hạn)        |  {test_cm[1][0]}  |  {test_cm[1][1]}")
    
    # In classification report
    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, test_preds, 
                              target_names=["Không đúng hạn", "Đúng hạn"]))
    
    # Lưu kết quả đánh giá để so sánh với các mô hình khác
    results_df = pd.DataFrame({
        'Model': ['CatBoost'],
        'AUC': [test_auc],
        'Accuracy': [test_accuracy],
        'F1 Score': [test_f1],
        'Precision': [test_precision],
        'Recall': [test_recall]
    })
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs('ml/catboost', exist_ok=True)
    
    # Lưu kết quả vào file CSV
    results_df.to_csv('ml/catboost/catboost_results.csv', index=False)
    # Lưu một bản sao ở thư mục gốc ml cho tương thích ngược
    results_df.to_csv('ml/catboost_results.csv', index=False)
    print("\nĐã lưu kết quả đánh giá vào ml/catboost/catboost_results.csv")
    
    # Trả về kết quả đánh giá và dự đoán
    evaluation_results = {
        'train': {
            'accuracy': train_accuracy,
            'f1': train_f1,
            'precision': train_precision,
            'recall': train_recall,
            'auc': train_auc,
            'predictions': train_preds,
            'probabilities': train_probs
        },
        'test': {
            'accuracy': test_accuracy,
            'f1': test_f1,
            'precision': test_precision,
            'recall': test_recall,
            'auc': test_auc,
            'predictions': test_preds,
            'probabilities': test_probs,
            'confusion_matrix': test_cm
        }
    }
    
    return evaluation_results

def extract_feature_importance(model, X_train):
    """
    Trích xuất và hiển thị độ quan trọng của các đặc trưng
    """
    print("\n===== ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG =====")
    
    # Lấy độ quan trọng của các đặc trưng
    feature_importance = model.get_feature_importance()
    feature_names = X_train.columns
    
    # Tạo DataFrame cho độ quan trọng của các đặc trưng
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Lưu độ quan trọng vào file CSV để so sánh với các mô hình khác
    importance_df.to_csv('ml/catboost/catboost_feature_importance.csv', index=False)
    # Lưu một bản sao ở thư mục gốc ml cho tương thích ngược
    importance_df.to_csv('ml/catboost_feature_importance.csv', index=False)
    print("Đã lưu độ quan trọng của các đặc trưng vào ml/catboost/catboost_feature_importance.csv")
    
    # In ra top 10 đặc trưng quan trọng nhất
    print("\nTop 10 đặc trưng quan trọng nhất:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['Feature']}: {row['Importance']:.6f}")
    
    return importance_df

def plot_confusion_matrix(confusion_matrix_data):
    """
    Vẽ confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix_data, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=["Không đúng hạn", "Đúng hạn"],
        yticklabels=["Không đúng hạn", "Đúng hạn"]
    )
    plt.xlabel('Giá trị thực tế')
    plt.ylabel('Giá trị dự đoán')
    plt.title('Confusion Matrix - CatBoost')
    plt.tight_layout()
    plt.savefig('ml/catboost/catboost_confusion_matrix.png', dpi=300)
    plt.close()
    
    print("Đã lưu confusion matrix vào ml/catboost/catboost_confusion_matrix.png")

def plot_feature_importance(importance_df):
    """
    Vẽ biểu đồ độ quan trọng của các đặc trưng
    """
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    
    # Vẽ biểu đồ
    sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
    plt.title('Top 15 đặc trưng quan trọng nhất - CatBoost', fontsize=14)
    plt.xlabel('Độ quan trọng', fontsize=12)
    plt.ylabel('Đặc trưng', fontsize=12)
    plt.tight_layout()
    plt.savefig('ml/catboost/catboost_feature_importance.png', dpi=300)
    plt.close()
    
    print("Đã lưu biểu đồ độ quan trọng vào ml/catboost/catboost_feature_importance.png")

def plot_roc_curve(y_test, y_probs):
    """
    Vẽ đường cong ROC
    """
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = sk_auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - CatBoost')
    plt.legend(loc='lower right')
    plt.savefig('ml/catboost/catboost_roc_curve.png', dpi=300)
    plt.close()
    
    print("Đã lưu đường cong ROC vào ml/catboost/catboost_roc_curve.png")

def plot_performance_metrics(evaluation_results):
    """
    Vẽ biểu đồ hiển thị các chỉ số hiệu suất
    """
    metrics = ['AUC', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
    train_values = [
        evaluation_results['train']['auc'],
        evaluation_results['train']['accuracy'],
        evaluation_results['train']['f1'],
        evaluation_results['train']['precision'],
        evaluation_results['train']['recall']
    ]
    
    test_values = [
        evaluation_results['test']['auc'],
        evaluation_results['test']['accuracy'],
        evaluation_results['test']['f1'],
        evaluation_results['test']['precision'],
        evaluation_results['test']['recall']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, train_values, width, label='Train')
    rects2 = ax.bar(x + width/2, test_values, width, label='Test')
    
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score')
    ax.set_title('Chỉ số hiệu suất của mô hình CatBoost')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Thêm giá trị lên đỉnh thanh
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('ml/catboost/catboost_performance_metrics.png', dpi=300)
    plt.close()
    
    print("Đã lưu biểu đồ chỉ số hiệu suất vào ml/catboost/catboost_performance_metrics.png")

def save_model(model, model_path):
    """
    Lưu mô hình đã huấn luyện
    """
    print(f"\nĐang lưu mô hình vào {model_path}...")
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model.save_model(model_path)
    print(f"Đã lưu mô hình thành công!")

def main():
    """
    Hàm chính để thực hiện toàn bộ quá trình
    """
    # Đọc dữ liệu
    train_df, test_df = load_data()
    
    # Chuẩn bị features
    X_train, y_train, X_test, y_test, categorical_cols = prepare_features(train_df, test_df)
    
    # Huấn luyện mô hình CatBoost
    model = train_catboost_model(X_train, y_train, X_test, y_test, categorical_cols)
    
    # Đánh giá mô hình
    evaluation_results = evaluate_model(model, X_train, y_train, X_test, y_test, categorical_cols)
    
    # Trích xuất độ quan trọng của các đặc trưng
    importance_df = extract_feature_importance(model, X_train)
    
    # Vẽ biểu đồ confusion matrix
    plot_confusion_matrix(evaluation_results['test']['confusion_matrix'])
    
    # Vẽ biểu đồ độ quan trọng của các đặc trưng
    plot_feature_importance(importance_df)
    
    # Vẽ đường cong ROC
    plot_roc_curve(y_test, evaluation_results['test']['probabilities'])
    
    # Vẽ biểu đồ hiển thị các chỉ số hiệu suất
    plot_performance_metrics(evaluation_results)
    
    # Lưu mô hình
    save_model(model, "ml/catboost/catboost_model.cbm")
    
    print("\nQuá trình huấn luyện và đánh giá đã hoàn thành!")

if __name__ == "__main__":
    main()
