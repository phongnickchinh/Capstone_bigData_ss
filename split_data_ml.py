import csv
import random
import os

# Set random seed for reproducibility
random.seed(42)

def split_data(input_file, train_file, test_file, train_size=9999, test_size=1000):
    print(f"Reading the original dataset: {input_file}")
    
    # Read all data into memory
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Get header
        data = list(reader)    # Read all rows
    
    total_records = len(data)
    print(f"Total records in the dataset: {total_records}")
    
    # Shuffle the data
    random.shuffle(data)
    
    # Split into train and test sets
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    
    # Write train data
    with open(train_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Write header
        writer.writerows(train_data)  # Write data
        
    # Write test data
    with open(test_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Write header
        writer.writerows(test_data)  # Write data
    
    print(f"Training data size: {len(train_data)} records")
    print(f"Test data size: {len(test_data)} records")

# Run the split
input_file = 'Train_cleaned.csv'
train_file = 'Train_ML.csv'
test_file = 'Test_ML.csv'

split_data(input_file, train_file, test_file)

print("Data split complete. Files created:")
print(f"1. {train_file} - 9999 records for training")
print(f"2. {test_file} - 1000 records for testing")
