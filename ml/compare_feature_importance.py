import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

def init_spark():
    """
    Khởi tạo SparkSession
    """
    spark = SparkSession.builder \
        .appName("Feature Importance Analysis") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def compare_feature_importance():
    """
    So sánh độ quan trọng của các đặc trưng từ Random Forest và phân tích tương quan
    """
    print("Đang so sánh độ quan trọng của các đặc trưng...")
    
    # Đọc ma trận tương quan từ phân tích trước đó
    # Kiểm tra xem có file train_ml không
    train_ml_path = 'ml/Train_ML.csv'
    train_cleaned_path = 'Train_cleaned.csv'
    
    if os.path.exists(train_ml_path):
        pandas_df = pd.read_csv(train_ml_path)
        print("Đã đọc dữ liệu từ Train_ML.csv")
    else:
        pandas_df = pd.read_csv(train_cleaned_path)
        print("Đã đọc dữ liệu từ Train_cleaned.csv")
    
    # Loại bỏ cột ID nếu có
    if 'ID' in pandas_df.columns:
        pandas_df = pandas_df.drop(columns=['ID'])
    
    # Chuyển đổi các biến phân loại thành số
    categorical_cols = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]
    
    for col in categorical_cols:
        if col in pandas_df.columns:
            pandas_df[col] = pandas_df[col].astype('category').cat.codes
    
    # Tính toán ma trận tương quan
    correlation_matrix = pandas_df.corr()
    target_correlations = correlation_matrix['Reached.on.Time_Y.N'].sort_values(ascending=False)
    
    # Loại bỏ tương quan của biến mục tiêu với chính nó
    target_correlations = target_correlations[target_correlations.index != 'Reached.on.Time_Y.N']
    
    # Tạo DataFrame cho tương quan
    corr_df = pd.DataFrame({'Feature': target_correlations.index, 'Correlation': target_correlations.values})
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    
    # Đọc độ quan trọng của các đặc trưng từ Random Forest (nếu đã chạy trước đó)
    try:
        spark = init_spark()
        model_path = "ml/rf_model"
        
        if os.path.exists(model_path):
            # Đọc mô hình đã huấn luyện
            model = PipelineModel.load(model_path)
            
            # Lấy RF model từ pipeline
            rf_model = model.stages[-1]
            
            # Lấy Vector Assembler để biết thứ tự các đặc trưng
            assembler = model.stages[-2]
            feature_names = assembler.getInputCols()
            
            # Lấy độ quan trọng của các đặc trưng
            importances = rf_model.featureImportances.toArray()
            
            # Tạo DataFrame
            feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
            
            # So sánh top 10 đặc trưng quan trọng nhất từ cả hai phương pháp
            print("\n===== TOP 10 ĐẶC TRƯNG QUAN TRỌNG - RANDOM FOREST =====")
            for i, (feature, importance) in enumerate(zip(feature_importance_df['Feature'][:10], 
                                                       feature_importance_df['Importance'][:10]), 1):
                print(f"{i}. {feature}: {importance:.6f}")
            
            print("\n===== TOP 10 ĐẶC TRƯNG QUAN TRỌNG - TƯƠNG QUAN PEARSON =====")
            for i, (feature, corr) in enumerate(zip(corr_df['Feature'][:10], 
                                                 corr_df['Correlation'][:10]), 1):
                print(f"{i}. {feature}: {corr:.6f}")
            
            # Tạo biểu đồ so sánh
            # Chỉ chọn các feature chung để so sánh (các biến số nguyên thủy)
            numeric_features = [col for col in corr_df['Feature'] if not any(col.endswith(suffix) for suffix in ["_index", "_encoded"])]
            
            # Tạo DataFrame cho so sánh
            compare_df = pd.DataFrame({'Feature': numeric_features})
            compare_df = compare_df.merge(corr_df[['Feature', 'Abs_Correlation']], on='Feature', how='left')
            compare_df = compare_df.rename(columns={'Abs_Correlation': 'Correlation_Abs'})
            
            # Tìm tương ứng trong feature importance
            for idx, row in compare_df.iterrows():
                feature = row['Feature']
                if feature in feature_importance_df['Feature'].values:
                    importance = feature_importance_df.loc[feature_importance_df['Feature'] == feature, 'Importance'].iloc[0]
                else:
                    importance = 0
                compare_df.at[idx, 'RF_Importance'] = importance
            
            # Chỉ lấy top features để hiển thị
            compare_df = compare_df.sort_values('RF_Importance', ascending=False).head(15)
            
            # Vẽ biểu đồ so sánh
            plt.figure(figsize=(12, 8))
            
            # Chuẩn hóa giá trị để có thể so sánh
            compare_df['Correlation_Abs_Norm'] = compare_df['Correlation_Abs'] / compare_df['Correlation_Abs'].max()
            compare_df['RF_Importance_Norm'] = compare_df['RF_Importance'] / compare_df['RF_Importance'].max()
            
            x = np.arange(len(compare_df))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.bar(x - width/2, compare_df['RF_Importance_Norm'], width, label='Random Forest')
            ax.bar(x + width/2, compare_df['Correlation_Abs_Norm'], width, label='Pearson Correlation')
            
            ax.set_xlabel('Đặc trưng')
            ax.set_ylabel('Độ quan trọng (đã chuẩn hóa)')
            ax.set_title('So sánh độ quan trọng của các đặc trưng')
            ax.set_xticks(x)
            ax.set_xticklabels(compare_df['Feature'], rotation=45, ha='right')
            ax.legend()
            
            fig.tight_layout()
            plt.savefig('ml/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("\nĐã lưu biểu đồ so sánh vào ml/feature_importance_comparison.png")
        else:
            print("\nKhông tìm thấy mô hình Random Forest. Vui lòng chạy huấn luyện mô hình trước.")
    
    except Exception as e:
        print(f"Lỗi khi so sánh độ quan trọng của các đặc trưng: {str(e)}")
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    compare_feature_importance()
