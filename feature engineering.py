import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# ✅ Update file name to match your folder listing
file_name = "EduPro Online Platform.xlsx - Users.csv"

# Create full path
file_path = os.path.join(script_dir, file_name)

print("Looking for file at:", file_path)

# ✅ Check if file exists before reading
if not os.path.exists(file_path):
    print("❌ ERROR: File not found!")
    print(f"👉 Make sure '{file_name}' is in the SAME folder as this script.")
    exit()

# ✅ Load file safely
try:
    # If it's truly a CSV file
    df = pd.read_csv(file_path)

    # If it's actually Excel despite the name, use:
    # df = pd.read_excel(file_path)

    print("✅ Data loaded successfully!\n")
    print(df.head())

except Exception as e:
    print("❌ ERROR while reading file:")
    print(e)
    exit()

# ✅ Example processing (optional)
print("\n🔧 Performing basic preprocessing...")

# Fill missing values
df.fillna(0, inplace=True)

# Encode categorical columns (if any)
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

print("✅ Preprocessing complete!")

# Save processed file
output_path = os.path.join(script_dir, "processed_users.csv")
df.to_csv(output_path, index=False)

print("📁 Processed file saved at:", output_path)
