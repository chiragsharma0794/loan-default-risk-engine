import pandas as pd
import lightgbm as lgb
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "loan_data_engineered.csv")

def check_leakage():
    print("Loading data to check for leakage...")
    df = pd.read_csv(INPUT_PATH)
    
    y = df['target']
    X = df.drop(columns=['target', 'loan_status'], errors='ignore')
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype('category')
        
    print("Training quick model to find leakage...")
    clf = lgb.LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    
    # Get feature importances by GAIN
    importances = clf.booster_.feature_importance(importance_type='gain')
    feature_names = X.columns
    
    # Create a dataframe and sort
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Gain': importances
    }).sort_values(by='Gain', ascending=False)
    
    print("\n🚨 TOP 10 FEATURES BY GAIN (Look for Leakage here) 🚨")
    print(importance_df.head(10).to_string(index=False))

if __name__ == "__main__":
    check_leakage()