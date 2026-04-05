import pandas as pd
import os

# Define paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "loan.csv")
SAMPLED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "loan_data_stratified_sample.csv")

TARGET_MAP = {
    "Fully Paid": 0,
    "Current": 0,
    "Charged Off": 1,
    "Default": 1
}

def create_time_balanced_sample(file_path, output_path, chunksize=100000):
    print("Starting time-balanced chunked data scan...")
    
    balanced_chunks = []
    
    # Process chunk by chunk to avoid memory issues
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize, low_memory=False)):
        print(f"Processing chunk {i+1}...")
        
        if 'loan_status' not in chunk.columns:
            continue
            
        chunk = chunk.copy()
        chunk['target'] = chunk['loan_status'].map(TARGET_MAP)
        valid_chunk = chunk.dropna(subset=['target'])
        
        defaults = valid_chunk[valid_chunk['target'] == 1]
        non_defaults = valid_chunk[valid_chunk['target'] == 0]
        
        # TIME-BALANCING: Take an equal number of non-defaults as defaults FROM THIS EXACT CHUNK
        n_defaults = len(defaults)
        if n_defaults > 0:
            if len(non_defaults) > n_defaults:
                non_defaults = non_defaults.sample(n_defaults, random_state=42)
            
            balanced_chunks.append(defaults)
            balanced_chunks.append(non_defaults)
            
        print(f"Chunk {i+1}: Kept {n_defaults} defaults and {len(non_defaults)} non-defaults.")
        
    print("Combining all time-balanced chunks...")
    final_sample = pd.concat(balanced_chunks)
    
    # Downsample to 100k rows total for fast modeling while keeping perfectly balanced
    if len(final_sample) > 100000:
        print("Downsampling to 100,000 rows...")
        final_sample = final_sample.groupby('target').sample(n=50000, random_state=42)
        
    final_sample = final_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_sample.to_csv(output_path, index=False)
    print(f"Saved time-balanced sample to {output_path}")
    print(f"Final shape: {final_sample.shape}")
    print(final_sample['target'].value_counts())

if __name__ == "__main__":
    create_time_balanced_sample(RAW_DATA_PATH, SAMPLED_DATA_PATH)