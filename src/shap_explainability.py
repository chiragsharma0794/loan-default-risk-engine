import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "loan_data_engineered.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

def generate_shap_plots():
    print("Loading data for SHAP analysis...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    
    y = df['target']
    X = df.drop(columns=['target', 'loan_status'], errors='ignore')
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype('category')
        
    print("Training model...")
    # Using 77 estimators since that was our best iteration in Step 4
    clf = lgb.LGBMClassifier(n_estimators=77, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    
    print("Calculating SHAP values on a sample of 10,000 rows (for speed)...")
    X_sample = X.sample(10000, random_state=42)
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle LightGBM binary classification output format
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    
    summary_path = os.path.join(OUTPUT_DIR, "shap_summary.png")
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved SHAP summary plot to {summary_path}")
    print("SHAP analysis complete! Check the outputs folder.")

if __name__ == "__main__":
    generate_shap_plots()