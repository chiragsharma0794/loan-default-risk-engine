import pandas as pd
import numpy as np
import os

# Define paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "loan_data_filtered_sample.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "loan_data_engineered.csv")

def engineer_features(df):
    print("Engineering features...")
    
    # 1. Log transforms for skewed financial variables
    if 'annual_inc' in df.columns:
        df['log_annual_inc'] = np.log1p(df['annual_inc'])
    if 'loan_amnt' in df.columns:
        df['log_loan_amnt'] = np.log1p(df['loan_amnt'])
        
    # 2. Income Ratios
    if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
        df['loan_to_inc_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)
        
    # 3. Credit Utilization Ratios
    if 'total_bal_ex_mort' in df.columns and 'annual_inc' in df.columns:
        df['total_bal_to_inc_ratio'] = df['total_bal_ex_mort'] / (df['annual_inc'] + 1)
        
    # 4. Account behavior aggregates
    if 'open_acc' in df.columns and 'total_acc' in df.columns:
        df['open_acc_ratio'] = df['open_acc'] / (df['total_acc'] + 1)
        
    # 5. Clean up employment length
    if 'emp_length' in df.columns:
        df['emp_length_num'] = df['emp_length'].str.extract(r'(d+)').astype(float)
        df['emp_length_num'] = df['emp_length_num'].fillna(0)
        
    print(f"Added {len([c for c in df.columns if c not in pd.read_csv(INPUT_PATH, nrows=0).columns])} new features.")
    return df

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    df_engineered = engineer_features(df)
    df_engineered.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved engineered data to {OUTPUT_PATH}")