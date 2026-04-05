import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import os

# Define paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "loan_data_engineered.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

def train_lgbm_model():
    print("Loading data...")
    df = pd.read_csv(INPUT_PATH)
    
    # Separate features and target
    y = df['target']
    X = df.drop(columns=['target', 'loan_status'], errors='ignore')
    
    # Handle categorical variables for LightGBM
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype('category')
        
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training LightGBM model...")
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # Evaluate
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nModel ROC-AUC: {auc:.4f}")
    
    # Feature Importance
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Changed importance_type to 'gain' to see which features actually improve the model most
    lgb.plot_importance(clf, max_num_features=20, figsize=(10, 8), importance_type='gain')
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, "lgbm_feature_importance.png")
    plt.savefig(output_file)
    print(f"Saved feature importance plot to {output_file}")
    
    return clf

if __name__ == "__main__":
    model = train_lgbm_model()