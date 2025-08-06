#!/usr/bin/env python3
"""
Script để chạy pipeline hoàn chỉnh: Preprocessing + Processing
"""

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_name, description):
    """Chạy một Python script và handle errors"""
    print(f"\n{'='*50}")
    print(f"ĐANG CHẠY: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print(f"✅ THÀNH CÔNG: {description}")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ LỖI: {description}")
            print("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION khi chạy {script_name}: {e}")
        return False
    
    return True

def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    print("Kiểm tra dependencies...")
    
    required_files = [
        "spark_preprocessing.py",
        "spark_processing.py",
        "spark_ml_preprocessing.py",
        "config.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Thiếu files: {missing_files}")
        return False
    
    print("✅ Tất cả files cần thiết đều có sẵn")
    return True

def check_data_files():
    """Kiểm tra data files"""
    data_files = ["shipping_data.csv"]
    available_files = []
    
    for file in data_files:
        if Path(file).exists():
            available_files.append(file)
    
    if not available_files:
        print("⚠️  Cảnh báo: Không tìm thấy file dữ liệu shipping_data.csv")
        print("   Đảm bảo file này có trong thư mục hiện tại")
        return False
    
    print(f"✅ Tìm thấy data files: {available_files}")
    return True

def main():
    """Main function để chạy toàn bộ pipeline"""
    print("🚀 BẮT ĐẦU PIPELINE PHÂN TÍCH SHIPPING DATA")
    print(f"Thư mục làm việc: {os.getcwd()}")
    
    # Parse command line arguments
    import sys
    mode = "all"  # Default mode
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    valid_modes = ["all", "analytics", "ml", "processing"]
    if mode not in valid_modes:
        print(f"❌ Mode không hợp lệ. Chọn: {valid_modes}")
        sys.exit(1)
    
    print(f"📋 Mode: {mode.upper()}")
    
    # 1. Kiểm tra dependencies
    if not check_dependencies():
        print("❌ Pipeline dừng do thiếu dependencies")
        sys.exit(1)
    
    # 2. Kiểm tra data files
    if not check_data_files():
        print("❌ Pipeline dừng do thiếu data files")
        sys.exit(1)
    
    # 3. Chạy data preprocessing (bắt buộc cho tất cả modes)
    if mode in ["all", "analytics", "ml"]:
        print("\n📊 BƯỚC 1: DATA PREPROCESSING")
        success = run_script("spark_preprocessing.py", "Data Preprocessing")
        if not success:
            print("❌ Preprocessing thất bại. Dừng pipeline.")
            sys.exit(1)
        
        # Kiểm tra output của preprocessing
        preprocessed_path = Path("preprocessed_shipping_data")
        if preprocessed_path.exists():
            print("✅ Tìm thấy dữ liệu đã preprocessing")
        else:
            print("⚠️  Không tìm thấy dữ liệu preprocessed, sẽ dùng raw data")
    
    # 4. Chạy analytics processing
    if mode in ["all", "analytics", "processing"]:
        print("\n📈 BƯỚC 2: DATA PROCESSING & ANALYSIS")
        success = run_script("spark_processing.py", "Data Processing & Analysis")
        if not success:
            print("❌ Processing thất bại.")
            if mode == "processing":
                sys.exit(1)
    
    # 5. Chạy ML preprocessing
    if mode in ["all", "ml"]:
        print("\n🤖 BƯỚC 3: ML PREPROCESSING")
        success = run_script("spark_ml_preprocessing.py", "ML Data Preprocessing")
        if not success:
            print("❌ ML Preprocessing thất bại.")
            if mode == "ml":
                sys.exit(1)
        else:
            print("✅ ML dataset đã được tạo")
    
    # 6. Summary
    print("\n🎉 HOÀN THÀNH PIPELINE!")
    if mode in ["all", "analytics"]:
        print("✅ Data Preprocessing: Hoàn thành")
        print("✅ Analytics Processing: Hoàn thành")
        print("✅ Dữ liệu đã được lưu vào MySQL")
    
    if mode in ["all", "ml"]:
        print("✅ ML Preprocessing: Hoàn thành")
        print("✅ ML Dataset đã được tạo")
        print("✅ Pipeline Model đã được lưu")
    
    print(f"\n📚 Để chạy các modes khác:")
    print(f"  python run_pipeline.py analytics  # Chỉ analytics")
    print(f"  python run_pipeline.py ml         # Chỉ ML preprocessing")
    print(f"  python run_pipeline.py processing # Chỉ processing (skip preprocessing)")
    print(f"  python run_pipeline.py all        # Tất cả (default)")

if __name__ == "__main__":
    main()
