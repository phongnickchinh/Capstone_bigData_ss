#!/usr/bin/env python3
"""
Script chạy trực tiếp trong WSL - không cần gọi wsl command
"""

import subprocess
import sys
import os
from datetime import datetime

class WSLSparkRunner:
    def __init__(self):
        self.project_path = "/mnt/p/coddd/Capstone_group4"
        
    def run_command(self, command, description="", show_output=True):
        """Chạy co        print("\n📋 MENU:")
        print("1. 🔍 Check installations")
        print("2. ☕ Setup Java 11") 
        print("3. ⚙️  Setup Hadoop & Spark")
        print("4. 🔧 Check MySQL JAR")
        print("5. 🚀 Start Hadoop only")
        print("6. ⚡ Run preprocessing only")
        print("7. 📊 Run analysis only") 
        print("8. 🔗 Run correlation analysis")
        print("9. 🎯 Run FULL PIPELINE")
        print("0. ❌ Exit") tiếp trong WSL"""
        if description:
            print(f"\n🔄 {description}")
        
        try:
            # Change to project directory first
            full_command = f"cd '{self.project_path}' && {command}"
            
            result = subprocess.run(full_command, 
                                  shell=True, 
                                  capture_output=not show_output,
                                  text=True,
                                  executable='/bin/bash')
            
            if result.returncode == 0:
                if not show_output and result.stdout:
                    print(result.stdout.strip())
                return True, result.stdout if not show_output else ""
            else:
                if not show_output:
                    print(f"❌ Error: {result.stderr}")
                return False, result.stderr if not show_output else ""
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False, str(e)
    
    def run_script(self, script_content, description=""):
        """Chạy multi-line script trực tiếp"""
        if description:
            print(f"\n🔄 {description}")
        
        try:
            # Create temp script file
            temp_script = f"/tmp/spark_script_{os.getpid()}.sh"
            
            # Write script to file
            with open(temp_script, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"cd '{self.project_path}'\n")
                f.write(script_content)
            
            # Make executable
            os.chmod(temp_script, 0o755)
            
            # Run script
            result = subprocess.run(['/bin/bash', temp_script], 
                                  capture_output=False, 
                                  text=True)
            
            # Cleanup
            try:
                os.remove(temp_script)
            except:
                pass
            
            return result.returncode == 0, ""
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False, str(e)
    
    def check_installations(self):
        """Kiểm tra các cài đặt cần thiết"""
        print("\n🔍 CHECKING INSTALLATIONS")
        print("-" * 40)
        
        # Check Java
        success, output = self.run_command("java -version 2>&1", show_output=False)
        if success and "11." in output:
            print("✅ Java 11: Found")
        else:
            print("❌ Java 11: Not found")
        
        # Check Hadoop
        success, _ = self.run_command("ls -la /home/phamp/hadoop/bin/hadoop", show_output=False)
        if success:
            print("✅ Hadoop: Found")
        else:
            print("❌ Hadoop: Not found")
        
        # Check Spark
        success, _ = self.run_command("ls -la /home/phamp/spark/bin/spark-submit", show_output=False)
        if success:
            print("✅ Spark: Found")
        else:
            print("❌ Spark: Not found")
        
        # Check PySpark
        success, _ = self.run_command("python3 -c 'import pyspark; print(pyspark.__version__)'", show_output=False)
        if success:
            print("✅ PySpark: Found")
        else:
            print("❌ PySpark: Not found")
        
        # Check MySQL JAR
        self.check_mysql_jar()
    
    def check_mysql_jar(self):
        """Kiểm tra MySQL JAR file"""
        print("\n🔍 CHECKING MYSQL JAR")
        print("-" * 30)

        jar_path = "/home/phamp/jars/mysql-connector-j-8.3.0.jar"
        success, _ = self.run_command(f"ls -la {jar_path}", show_output=False)
        if success:
            print("✅ MySQL JAR found")
            return True
        else:
            print("❌ MySQL JAR not found")
            return False
    
    def check_java(self):
        """Kiểm tra Java"""
        print("\n☕ KIỂM TRA JAVA")
        print("-" * 30)
        
        success, output = self.run_command("java -version 2>&1", show_output=False)
        if success and "11." in output:
            print(f"✅ Java 11 found")
            return True
        elif success:
            print(f"⚠️  Java found but not version 11")
            return False
        else:
            print("❌ Java not found")
            return False
    
    def setup_java(self):
        """Thiết lập Java 11"""
        print("\n📦 SETUP JAVA 11")
        print("-" * 30)
        
        commands = [
            ("sudo apt update", "Updating package list"),
            ("sudo apt install -y openjdk-11-jdk", "Installing Java 11"),
            ("echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc", "Setting JAVA_HOME"),
            ("echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc", "Adding Java to PATH")
        ]
        
        for cmd, desc in commands:
            success, _ = self.run_command(cmd, desc, show_output=False)
            if success:
                print(f"✅ {desc}")
            else:
                print(f"❌ Failed: {desc}")
                return False
        
        return True
    
    def setup_hadoop_spark(self):
        """Cài đặt Hadoop và Spark"""
        print("\n⚙️  SETUP HADOOP & SPARK")
        print("-" * 30)
        
        setup_script = '''
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Tạo thư mục
sudo mkdir -p /opt
cd /tmp

# Cài Hadoop nếu chưa có
if [ ! -d "/opt/hadoop" ]; then
    echo "Installing Hadoop..."
    wget -q https://archive.apache.org/dist/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
    sudo tar -xzf hadoop-3.3.6.tar.gz -C /opt
    sudo mv /opt/hadoop-3.3.6 /opt/hadoop
    sudo chown -R $USER:$USER /opt/hadoop
fi

# Cài Spark nếu chưa có
if [ ! -d "/opt/spark" ]; then
    echo "Installing Spark..."
    wget -q https://archive.apache.org/dist/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz
    sudo tar -xzf spark-3.4.1-bin-hadoop3.tgz -C /opt
    sudo mv /opt/spark-3.4.1-bin-hadoop3 /opt/spark
    sudo chown -R $USER:$USER /opt/spark
fi

# Setup environment
grep -q "HADOOP_HOME" ~/.bashrc || echo "export HADOOP_HOME=/opt/hadoop" >> ~/.bashrc
grep -q "SPARK_HOME" ~/.bashrc || echo "export SPARK_HOME=/opt/spark" >> ~/.bashrc
grep -q "HADOOP_HOME/bin" ~/.bashrc || echo "export PATH=\\$PATH:/opt/hadoop/bin:/opt/hadoop/sbin" >> ~/.bashrc
grep -q "SPARK_HOME/bin" ~/.bashrc || echo "export PATH=\\$PATH:/opt/spark/bin:/opt/spark/sbin" >> ~/.bashrc
grep -q "PYTHONPATH" ~/.bashrc || echo "export PYTHONPATH=/opt/spark/python:/opt/spark/python/lib/py4j-*.zip:\\$PYTHONPATH" >> ~/.bashrc

# Cấu hình Hadoop
sudo mkdir -p /opt/hadoop/etc/hadoop
cat > /tmp/core-site.xml << 'EOF'
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
</configuration>
EOF
sudo mv /tmp/core-site.xml /opt/hadoop/etc/hadoop/

echo "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64" >> /opt/hadoop/etc/hadoop/hadoop-env.sh

echo "Setup completed!"
'''
        
        success, _ = self.run_script(setup_script, "Setting up Hadoop & Spark")
        return success
    
    def start_hadoop(self):
        """Khởi động Hadoop"""
        print("\n🚀 STARTING HADOOP")
        print("-" * 30)
        
        start_script = '''
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export HADOOP_HOME=/home/phamp/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

# Source bashrc to get environment
source ~/.bashrc

# Check if Hadoop is already running
if jps | grep -q "NameNode\|DataNode"; then
    echo "✅ Hadoop services are already running:"
    jps | grep -E "(NameNode|DataNode|SecondaryNameNode|ResourceManager|NodeManager)"
else
    echo "Starting Hadoop services..."
    
    # Format namenode lần đầu nếu cần
    if [ ! -d "/tmp/hadoop-$USER" ]; then
        echo "Formatting namenode..."
        echo "Y" | $HADOOP_HOME/bin/hdfs namenode -format
    fi

    # Start DFS services
    $HADOOP_HOME/sbin/start-dfs.sh
    
    # Start YARN services  
    $HADOOP_HOME/sbin/start-yarn.sh
    
    # Wait for services to start
    echo "Waiting for services to start..."
    sleep 10
fi

# Ensure HDFS directories exist
echo "Setting up HDFS directories..."
$HADOOP_HOME/bin/hdfs dfs -mkdir -p /input 2>/dev/null || echo "Input directory already exists"
$HADOOP_HOME/bin/hdfs dfs -mkdir -p /output 2>/dev/null || echo "Output directory already exists"

# Upload data file
echo "Uploading data file..."
if $HADOOP_HOME/bin/hdfs dfs -test -e /input/Train.csv; then
    echo "✅ Train.csv already exists in HDFS"
else
    if [ -f "Train.csv" ]; then
        $HADOOP_HOME/bin/hdfs dfs -put Train.csv /input/
        echo "✅ Train.csv uploaded to HDFS"
    else
        echo "❌ Train.csv not found in current directory"
    fi
fi

echo "Hadoop setup completed!"
echo "Current running processes:"
jps
'''
        
        return self.run_script(start_script, "Starting Hadoop services")
    
    def run_spark_job(self, script_name):
        """Chạy Spark job"""
        print(f"\n⚡ RUNNING {script_name}")
        print("-" * 30)
        
        # Check MySQL JAR first
        if not self.check_mysql_jar():
            print("❌ MySQL JAR not found. Cannot continue.")
            return False
        
        spark_script = f'''
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export HADOOP_HOME=/home/phamp/hadoop
export SPARK_HOME=/home/phamp/spark
export PATH=$PATH:$HADOOP_HOME/bin:$SPARK_HOME/bin
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-*.zip:$PYTHONPATH

# Source bashrc
source ~/.bashrc

# Check if MySQL JAR exists
JAR_PATH="/home/phamp/jars/mysql-connector-j-8.3.0.jar"
if [ ! -f "$JAR_PATH" ]; then
    echo "❌ MySQL JAR not found at $JAR_PATH"
    exit 1
fi

echo "Running {script_name} with MySQL connector..."
echo "Using JAR: $JAR_PATH"
echo "SPARK_HOME: $SPARK_HOME"
echo "HADOOP_HOME: $HADOOP_HOME"

# Use spark-submit to ensure JAR is properly loaded with reduced logging
$SPARK_HOME/bin/spark-submit \\
    --jars $JAR_PATH \\
    --driver-class-path $JAR_PATH \\
    --conf spark.executor.extraClassPath=$JAR_PATH \\
    --conf spark.driver.extraClassPath=$JAR_PATH \\
    --conf spark.sql.warehouse.dir=/tmp/spark-warehouse \\
    --conf spark.sql.adaptive.enabled=true \\
    --conf spark.sql.adaptive.coalescePartitions.enabled=true \\
    --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:log4j.properties" \\
    --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:log4j.properties" \\
    {script_name} 2>/dev/null
'''
        
        return self.run_script(spark_script, f"Running {script_name}")
    
    def run_correlation_analysis(self):
        """Chạy correlation analysis với ML preprocessing"""
        print("\n🔗 CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Kiểm tra xem có preprocessed data không
        check_cmd = "ls -la preprocessed_data/*.csv 2>/dev/null | wc -l"
        success, output = self.run_command(check_cmd, "", show_output=False)
        
        if success and output.strip() == "0":
            print("⚠️  Không tìm thấy preprocessed data. Chạy preprocessing trước...")
            if not self.run_spark_job("spark_preprocessing.py"):
                print("❌ Preprocessing failed")
                return False
        
        # Chạy correlation analysis
        return self.run_spark_job("machine_learning/run_correlation_analysis.py")

    def full_pipeline(self):
        """Chạy full pipeline"""
        print("\n🎯 FULL PIPELINE")
        print("=" * 50)
        
        # Start Hadoop
        if not self.start_hadoop():
            print("❌ Failed to start Hadoop")
            return False
        
        # Run preprocessing
        if not self.run_spark_job("spark_preprocessing.py"):
            print("❌ Preprocessing failed")
            return False
        
        # Run analysis
        if not self.run_spark_job("analysis/shipping_analy.py"):
            print("❌ Analysis failed")
            return False
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        return True

def main():
    """Menu chính"""

    runner = WSLSparkRunner()
    
    print("🔥 WSL SPARK RUNNER - NATIVE WSL VERSION")
    print("=" * 50)
    
    while True:
        print("\n📋 MENU:")
        print("1. 🔍 Check installations")
        print("2. ☕ Setup Java 11") 
        print("3. ⚙️  Setup Hadoop & Spark")
        print("4. � Check MySQL JAR")
        print("5. �🚀 Start Hadoop only")
        print("6. ⚡ Run preprocessing only")
        print("7. 📊 Run analysis only") 
        print("8. 🎯 Run FULL PIPELINE")
        print("0. ❌ Exit")
        
        choice = input("\nChọn (0-9): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            runner.check_installations()
        elif choice == "2":
            runner.setup_java()
        elif choice == "3":
            runner.setup_hadoop_spark()
        elif choice == "4":
            runner.check_mysql_jar()
        elif choice == "5":
            runner.start_hadoop()
        elif choice == "6":
            runner.run_spark_job("spark_preprocessing.py")
        elif choice == "7":
            runner.run_spark_job("analysis/shipping_analy.py")
        elif choice == "8":
            runner.run_correlation_analysis()
        elif choice == "9":
            runner.full_pipeline()
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
