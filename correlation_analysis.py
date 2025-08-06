import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_correlation_heatmap():
    """
    Tạo heatmap hiển thị mức độ tương quan giữa các thuộc tính và 
    mức độ ảnh hưởng của chúng đến biến mục tiêu Reached.on.Time_Y.N
    """
    # Đọc dữ liệu train trực tiếp bằng Pandas thay vì PySpark
    print("Đang đọc dữ liệu training...")
    
    # Xác định đường dẫn file
    train_ml_path = '/mnt/p/coddd/Capstone_group4/Train_ML.csv'
    train_cleaned_path = '/mnt/p/coddd/Capstone_group4/Train_cleaned.csv'
    
    # Kiểm tra file Train_ML.csv tồn tại
    if os.path.exists(train_ml_path):
        pandas_df = pd.read_csv(train_ml_path)
        print("Đã đọc dữ liệu từ Train_ML.csv")
    else:
        # Nếu không tìm thấy, đọc từ file gốc
        pandas_df = pd.read_csv(train_cleaned_path)
        print("Đã đọc dữ liệu từ Train_cleaned.csv")
    
    # Loại bỏ cột ID nếu có
    if 'ID' in pandas_df.columns:
        pandas_df = pandas_df.drop(columns=['ID'])
    
    # Chuyển đổi các biến phân loại thành số
    categorical_cols = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]
    
    # Dùng label encoding thay vì one-hot encoding để giữ nguyên cấu trúc các cột
    for col in categorical_cols:
        if col in pandas_df.columns:
            pandas_df[col] = pandas_df[col].astype('category').cat.codes
    
    # Tính toán ma trận tương quan giữa các biến (không tách lẻ biến phân loại)
    correlation_matrix = pandas_df.corr()
    

    print("Đã tính toán ma trận tương quan.")

    # Hiển thị ma trận tương quan hoàn chỉnh
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Ma trận tương quan giữa các thuộc tính', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_matrix_full.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Loại bỏ các cột có tương quan <= |0.01| với biến mục tiêu
    low_corr_cols = correlation_matrix[correlation_matrix['Reached.on.Time_Y.N'].abs() <= 0.01].index
    correlation_matrix = correlation_matrix.drop(columns=low_corr_cols, index=low_corr_cols)
    correlation_matrix = correlation_matrix.round(2)
    print("Đã loại bỏ các cột có tương quan <= |0.01| với biến mục tiêu.")
    # Hiển thị ma trận tương quan đã lọc
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Ma trận tương quan đã lọc bỏ các cột có tương quan thấp', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_matrix_filtered.png', dpi=300, bbox_inches='tight')
    plt.close()


    # Hiển thị mức độ ảnh hưởng của các thuộc tính đến biến mục tiêu
    target_correlations = correlation_matrix['Reached.on.Time_Y.N'].sort_values(ascending=False)
    
    # Loại bỏ tương quan của biến mục tiêu với chính nó
    target_correlations = target_correlations[target_correlations.index != 'Reached.on.Time_Y.N']
    
    # Vẽ heatmap cho mức độ ảnh hưởng
    plt.figure(figsize=(10, 12))
    sns.heatmap(
        target_correlations.to_frame(), 
        annot=True, 
        cmap='coolwarm', 
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={"label": "Hệ số tương quan Pearson"}
    )
    plt.title('Mức độ ảnh hưởng của các thuộc tính đến Reached.on.Time_Y.N', fontsize=16)
    plt.tight_layout()
    plt.savefig('feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # In ra top 5 thuộc tính ảnh hưởng nhất (cả tích cực và tiêu cực)
    print("\nTop 5 thuộc tính có ảnh hưởng tích cực nhất đến giao hàng đúng hạn:")
    print(target_correlations.head(5))
    
    print("\nTop 5 thuộc tính có ảnh hưởng tiêu cực nhất đến giao hàng đúng hạn:")
    print(target_correlations.tail(5))
    
    print("\nĐã lưu heatmap vào các file: correlation_matrix_full.png và feature_importance_heatmap.png")

if __name__ == "__main__":
    create_correlation_heatmap()
