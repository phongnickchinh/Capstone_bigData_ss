"""
Statistical Visualization Script
Tạo biểu đồ trực quan cho các thống kê mô tả
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# Thiết lập style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_statistical_visualizations():
    """Tạo các biểu đồ thống kê từ dữ liệu"""
    
    print("📊 CREATING STATISTICAL VISUALIZATIONS")
    print("=" * 60)
    
    # Dữ liệu thống kê từ bảng
    stats_data = {
        'Column': ['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product', 
                  'Prior_purchases', 'Discount_offered', 'Weight_in_gms'],
        'Mean': [4.05, 2.99, 210.20, 3.57, 13.37, 3634.02],
        'Median': [4.00, 3.00, 214.00, 3.00, 7.00, 4149.00],
        'Std_Dev': [1.14, 1.41, 48.06, 1.52, 16.21, 1635.38],
        'Min': [2, 1, 96, 2, 1, 1001],
        'Max': [7, 5, 310, 10, 65, 7846]
    }
    
    df = pd.DataFrame(stats_data)
    
    # Tạo figure với nhiều subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. So sánh Mean vs Median
    plt.subplot(4, 2, 1)
    x_pos = np.arange(len(df['Column']))
    width = 0.35
    
    plt.bar(x_pos - width/2, df['Mean'], width, label='Mean', color='skyblue', alpha=0.8)
    plt.bar(x_pos + width/2, df['Median'], width, label='Median', color='lightcoral', alpha=0.8)
    
    plt.xlabel('Variables', fontweight='bold')
    plt.ylabel('Values', fontweight='bold')
    plt.title('Mean vs Median Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, df['Column'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 2. Standard Deviation
    plt.subplot(4, 2, 2)
    bars = plt.bar(df['Column'], df['Std_Dev'], color='lightgreen', alpha=0.7)
    plt.xlabel('Variables', fontweight='bold')
    plt.ylabel('Standard Deviation', fontweight='bold')
    plt.title('Standard Deviation by Variable', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Thêm giá trị trên mỗi bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Range (Max - Min)
    plt.subplot(4, 2, 3)
    ranges = np.array(df['Max']) - np.array(df['Min'])
    bars = plt.bar(df['Column'], ranges, color='orange', alpha=0.7)
    plt.xlabel('Variables', fontweight='bold')
    plt.ylabel('Range (Max - Min)', fontweight='bold')
    plt.title('Data Range by Variable', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Thêm giá trị trên mỗi bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Coefficient of Variation (CV = Std/Mean)
    plt.subplot(4, 2, 4)
    cv = np.array(df['Std_Dev']) / np.array(df['Mean'])
    bars = plt.bar(df['Column'], cv, color='purple', alpha=0.7)
    plt.xlabel('Variables', fontweight='bold')
    plt.ylabel('Coefficient of Variation', fontweight='bold')
    plt.title('Coefficient of Variation (Std Dev / Mean)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Thêm giá trị trên mỗi bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Min-Max normalization visualization
    plt.subplot(4, 2, 5)
    
    # Normalize các giá trị để có thể so sánh trên cùng một biểu đồ
    normalized_data = {}
    for col in ['Mean', 'Median', 'Min', 'Max']:
        values = np.array(df[col])
        # Min-Max scaling để đưa về [0, 1]
        normalized_data[col] = (values - values.min()) / (values.max() - values.min())
    
    x_pos = np.arange(len(df['Column']))
    width = 0.2
    
    plt.bar(x_pos - 1.5*width, normalized_data['Min'], width, label='Min', color='red', alpha=0.7)
    plt.bar(x_pos - 0.5*width, normalized_data['Mean'], width, label='Mean', color='blue', alpha=0.7)
    plt.bar(x_pos + 0.5*width, normalized_data['Median'], width, label='Median', color='green', alpha=0.7)
    plt.bar(x_pos + 1.5*width, normalized_data['Max'], width, label='Max', color='orange', alpha=0.7)
    
    plt.xlabel('Variables', fontweight='bold')
    plt.ylabel('Normalized Values (0-1)', fontweight='bold')
    plt.title('Normalized Statistical Measures', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, df['Column'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Box plot style visualization
    plt.subplot(4, 2, 6)
    
    # Tạo data cho box plot style
    box_data = []
    labels = []
    
    for i, row in df.iterrows():
        # Simulate distribution based on mean, std, min, max
        # Tạo dữ liệu mô phỏng để vẽ box plot
        mean_val = row['Mean']
        std_val = row['Std_Dev']
        min_val = row['Min']
        max_val = row['Max']
        
        # Tạo distribution
        data_points = np.random.normal(mean_val, std_val, 1000)
        # Clip to min-max range
        data_points = np.clip(data_points, min_val, max_val)
        
        box_data.append(data_points)
        labels.append(row['Column'])
    
    plt.boxplot(box_data, labels=labels)
    plt.xlabel('Variables', fontweight='bold')
    plt.ylabel('Values', fontweight='bold')
    plt.title('Simulated Distribution Box Plots', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 7. Heatmap của các thống kê
    plt.subplot(4, 2, 7)
    
    # Tạo correlation-style heatmap cho các thống kê
    stats_matrix = df[['Mean', 'Median', 'Std_Dev', 'Min', 'Max']].T
    stats_matrix.columns = df['Column']
    
    # Normalize từng hàng để có thể so sánh
    stats_normalized = stats_matrix.div(stats_matrix.max(axis=1), axis=0)
    
    sns.heatmap(stats_normalized, annot=True, cmap='YlOrRd', fmt='.2f', 
                cbar_kws={'label': 'Normalized Value'})
    plt.title('Statistical Measures Heatmap (Normalized)', fontsize=14, fontweight='bold')
    plt.ylabel('Statistical Measures', fontweight='bold')
    plt.xlabel('Variables', fontweight='bold')
    
    # 8. Radar chart style comparison
    plt.subplot(4, 2, 8)
    
    # Tạo radar chart để so sánh các biến
    angles = np.linspace(0, 2 * np.pi, len(df), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Normalize data for radar chart
    mean_norm = (np.array(df['Mean']) - np.array(df['Mean']).min()) / (np.array(df['Mean']).max() - np.array(df['Mean']).min())
    std_norm = (np.array(df['Std_Dev']) - np.array(df['Std_Dev']).min()) / (np.array(df['Std_Dev']).max() - np.array(df['Std_Dev']).min())
    
    mean_norm = np.concatenate((mean_norm, [mean_norm[0]]))
    std_norm = np.concatenate((std_norm, [std_norm[0]]))
    
    ax = plt.subplot(4, 2, 8, projection='polar')
    ax.plot(angles, mean_norm, 'o-', linewidth=2, label='Mean (normalized)', color='blue')
    ax.fill(angles, mean_norm, alpha=0.25, color='blue')
    ax.plot(angles, std_norm, 'o-', linewidth=2, label='Std Dev (normalized)', color='red')
    ax.fill(angles, std_norm, alpha=0.25, color='red')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([col.replace('_', '\n') for col in df['Column']])
    ax.set_ylim(0, 1)
    ax.set_title('Mean vs Standard Deviation\n(Radar Chart)', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"statistical_visualizations_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Biểu đồ đã được lưu: {filename}")
    
    # Show plot
    plt.show()
    
    return filename

def create_individual_charts():
    """Tạo các biểu đồ riêng lẻ cho từng biến"""
    
    print("\n📈 CREATING INDIVIDUAL VARIABLE CHARTS")
    print("=" * 50)
    
    # Dữ liệu chi tiết cho từng biến
    variables_data = {
        'Customer_care_calls': {
            'mean': 4.05, 'median': 4.00, 'std': 1.14, 'min': 2, 'max': 7,
            'description': 'Number of customer service calls',
            'unit': 'calls'
        },
        'Customer_rating': {
            'mean': 2.99, 'median': 3.00, 'std': 1.41, 'min': 1, 'max': 5,
            'description': 'Customer satisfaction rating',
            'unit': 'stars'
        },
        'Cost_of_the_Product': {
            'mean': 210.20, 'median': 214.00, 'std': 48.06, 'min': 96, 'max': 310,
            'description': 'Product cost in currency',
            'unit': 'currency units'
        },
        'Prior_purchases': {
            'mean': 3.57, 'median': 3.00, 'std': 1.52, 'min': 2, 'max': 10,
            'description': 'Number of previous purchases',
            'unit': 'purchases'
        },
        'Discount_offered': {
            'mean': 13.37, 'median': 7.00, 'std': 16.21, 'min': 1, 'max': 65,
            'description': 'Discount percentage offered',
            'unit': '%'
        },
        'Weight_in_gms': {
            'mean': 3634.02, 'median': 4149.00, 'std': 1635.38, 'min': 1001, 'max': 7846,
            'description': 'Package weight',
            'unit': 'grams'
        }
    }
    
    # Tạo figure cho individual charts
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, (var_name, data) in enumerate(variables_data.items()):
        ax = axes[i]
        
        # Tạo simulated distribution
        np.random.seed(42 + i)  # For reproducibility
        samples = np.random.normal(data['mean'], data['std'], 1000)
        samples = np.clip(samples, data['min'], data['max'])
        
        # Histogram with statistics overlay
        ax.hist(samples, bins=30, alpha=0.7, color=plt.cm.Set3(i), density=True, edgecolor='black')
        
        # Add vertical lines for mean and median
        ax.axvline(data['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {data['mean']:.2f}")
        ax.axvline(data['median'], color='blue', linestyle='--', linewidth=2, label=f"Median: {data['median']:.2f}")
        
        # Add text box with statistics
        stats_text = f"Mean: {data['mean']:.2f}\nMedian: {data['median']:.2f}\nStd: {data['std']:.2f}\nRange: {data['min']}-{data['max']}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title(f"{var_name.replace('_', ' ').title()}\n({data['description']})", fontweight='bold', fontsize=10)
        ax.set_xlabel(f"Values ({data['unit']})", fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    # Lưu biểu đồ individual
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_individual = f"individual_variable_charts_{timestamp}.png"
    plt.savefig(filename_individual, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Biểu đồ cá nhân đã được lưu: {filename_individual}")
    
    plt.show()
    
    return filename_individual

def create_summary_table():
    """Tạo bảng tóm tắt với formatting đẹp"""
    
    print("\n📋 CREATING FORMATTED SUMMARY TABLE")
    print("=" * 50)
    
    # Dữ liệu
    stats_data = {
        'Variable': ['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product', 
                    'Prior_purchases', 'Discount_offered', 'Weight_in_gms'],
        'Mean': [4.05, 2.99, 210.20, 3.57, 13.37, 3634.02],
        'Median': [4.00, 3.00, 214.00, 3.00, 7.00, 4149.00],
        'Std Dev': [1.14, 1.41, 48.06, 1.52, 16.21, 1635.38],
        'Min': [2, 1, 96, 2, 1, 1001],
        'Max': [7, 5, 310, 10, 65, 7846]
    }
    
    df = pd.DataFrame(stats_data)
    
    # Tạo figure cho bảng
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Tạo bảng
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    
    # Format bảng
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Màu header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Màu alternating rows
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title('Statistical Summary of Numerical Variables', fontsize=16, fontweight='bold', pad=20)
    
    # Lưu bảng
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_table = f"summary_table_{timestamp}.png"
    plt.savefig(filename_table, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Bảng tóm tắt đã được lưu: {filename_table}")
    
    plt.show()
    
    return filename_table

def main():
    """Main function"""
    
    print("🎨 STATISTICAL VISUALIZATION GENERATOR")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now()}")
    
    try:
        # Tạo các biểu đồ chính
        main_chart = create_statistical_visualizations()
        
        # Tạo biểu đồ cá nhân
        individual_charts = create_individual_charts()
        
        # Tạo bảng tóm tắt
        summary_table = create_summary_table()
        
        print(f"\n✅ TẤT CẢ BIỂU ĐỒ ĐÃ ĐƯỢC TẠO THÀNH CÔNG!")
        print(f"📁 Files generated:")
        print(f"   • {main_chart}")
        print(f"   • {individual_charts}")
        print(f"   • {summary_table}")
        
        print(f"\n📊 Biểu đồ bao gồm:")
        print("   • So sánh Mean vs Median")
        print("   • Biểu đồ Standard Deviation")
        print("   • Biểu đồ Range (Max-Min)")
        print("   • Coefficient of Variation")
        print("   • Normalized comparison")
        print("   • Simulated box plots")
        print("   • Statistical heatmap")
        print("   • Radar chart comparison")
        print("   • Individual variable distributions")
        print("   • Formatted summary table")
        
    except Exception as e:
        print(f"❌ Lỗi khi tạo biểu đồ: {e}")
        print("💡 Đảm bảo các thư viện matplotlib, seaborn, pandas, numpy đã được cài đặt")
    
    print(f"\n🕒 Completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
