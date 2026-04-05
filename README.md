# 🏦 Loan Default Risk Engine
**Built for HackRush '26**

## 📌 Overview
The Loan Default Risk Engine is a machine learning pipeline designed to predict the probability of a borrower defaulting on a loan *at the time of origination*. 

Unlike many naive models that suffer from "data leakage" (using future data to predict the past), this project heavily focuses on **Data Integrity** and **Explainable AI (XAI)**.

## 🚀 Key Features & Methodology

**1. Eradicating Temporal Bias (Time Leakage)**
Initial models scored an unrealistic 0.9975 ROC-AUC. Deep debugging revealed a temporal bias: defaults were being sampled from older loans (2015-2017) and non-defaults from newer loans (2018). We implemented a **Time-Balanced Stratified Sampling** algorithm to balance classes *within* specific time chunks, forcing the model to learn actual risk factors rather than timelines.

**2. Financial Feature Engineering**
Engineered custom, domain-specific features to capture borrower risk, including:
* `loan_to_inc_ratio`: Loan amount relative to annual income.
* Log transforms on heavily skewed financial metrics.

**3. Gradient Boosting (LightGBM)**
Trained a highly optimized LightGBM classifier on the cleaned, engineered dataset, achieving a realistic, production-ready **ROC-AUC of 0.6885** (standard for pre-loan origination data).

**4. Explainable AI (SHAP)**
In the financial sector, models cannot be black boxes. We utilized SHAP (SHapley Additive exPlanations) to prove the model learned sound financial logic. The SHAP summary plots confirm that high Interest Rates (`int_rate`), high Debt-to-Income (`dti`), and high Loan-to-Income ratios are the primary drivers of default predictions.

## 📂 Project Structure
```text
loan_default_project/
├── data/                  # Ignored in Git (Raw, Interim, Processed CSVs)
├── outputs/               # Model outputs and SHAP summary plots
├── src/
│   ├── stratified_sampling.py   # Time-balanced chunk sampling
│   ├── feature_screening.py     # Drops leaky/future features
│   ├── feature_engineering.py   # Creates financial ratios
│   ├── lgbm_model.py            # Trains the LightGBM classifier
│   ├── check_leakage.py         # Debugging script using Gain importance
│   └── shap_explainability.py   # Generates XAI visualizations
├── .gitignore
└── README.md