#!/usr/bin/env python3
"""
Script chạy Spark job trên Windows (không dùng WSL)
Đơn giản hóa cho Windows thuần túy
"""

import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name, description):
    """Chạy Python script"""
    print(f"\n{'='*30}")
    print(f"🔄 {description}")
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*30}")
    
    python_exe = r"P:\coddd\Capstone_group4\.venv\Scripts\python.exe"
    
    try:
        result = subprocess.run([python_exe, script_name], 
                              check=True)
        print(f"✅ {description} - SUCCESS")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (code {e.returncode})")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_files():
    """Kiểm tra file cần thiết"""
    required_files = [
        "spark_preprocessing.py",
        "shipping_analy.py", 
        "config.py",
        "mysql-connector-j-8.3.0.jar",
        "spark_utils.py",
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print("❌ Missing files:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    print("✅ All required files present")
    return True

def main():
    """Hàm chính - Windows Pipeline"""
    print("🔥 WINDOWS SPARK PIPELINE")
    print("=" * 40)
    print(f"📁 Directory: {os.getcwd()}")
    print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Kiểm tra files
    if not check_files():
        return False
    
    # Chạy preprocessing
    if not run_script("spark_preprocessing.py", "Data Preprocessing"):
        return False
    
    # Chạy analysis
    if not run_script("shipping_analy.py", "Shipping Analysis"):
        return False
    
    print(f"\n🎉 PIPELINE COMPLETED!")
    print(f"⏰ End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return True

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
