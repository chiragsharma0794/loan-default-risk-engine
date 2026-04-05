import pandas as pd
import os

# Define paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "loan_data_stratified_sample.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "loan_data_filtered_sample.csv")

def screen_features(df):
    print(f"Initial shape: {df.shape}")
    
    # 1. Remove identifiers
    identifiers = ['id', 'member_id', 'url', 'emp_title', 'title', 'desc']
    df = df.drop(columns=[c for c in identifiers if c in df.columns])
    
    # 2. Remove leakage columns (post-loan features)
    leakage_cols = [
        'funded_amnt_inv', 'issue_d', 'out_prncp', 'out_prncp_inv',
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
        'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
        'debt_settlement_flag', 'debt_settlement_flag_date', 'settlement_status',
        'settlement_date', 'settlement_amount', 'settlement_percentage', 'settlement_term',
        'last_fico_range_high', 'last_fico_range_low' # <-- ADDED THESE SNEAKY LEAKAGE COLUMNS
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    
    # 3. Remove hardship columns
    hardship_cols = [c for c in df.columns if 'hardship' in c]
    df = df.drop(columns=hardship_cols)
    
    # 4. Remove columns with > 80% missing values
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.8].index
    df = df.drop(columns=cols_to_drop)
    
    print(f"Final shape after screening: {df.shape}")
    return df

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df_filtered = screen_features(df)
    df_filtered.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved filtered data to {OUTPUT_PATH}")