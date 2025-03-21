#!/usr/bin/env python3
"""
employee_production.py

Version: 1.0
Description:
    This script implements the production pipeline for employee turnover prediction.
    It loads raw employee and request data, calls the preprocessing module,
    trains an XGBoost classifier with hyperparameter tuning, predicts employee turnover,
    and generates local interpretability explanations using SHAP.

Usage:
    python employee_production.py --employee_file path/to/employee_data.xlsx --request_file path/to/request_data.csv
"""

import argparse
import pandas as pd
import numpy as np
import time
from datetime import datetime
from employee_preprocess import preprocess_employee_data
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, recall_score
import shap
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(1)

def train_model(X_train, y_train, scoring, cv=3):
    """
    Train an XGBoost classifier using GridSearchCV with specified hyperparameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        scoring (str): Scoring metric to optimize (e.g. 'recall').
        cv (int): Number of cross-validation folds.

    Returns:
        tuple: (best_model, best_params, best_score)
    """
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [1]
    }
    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, verbose=1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_

def main(args):
    start_time = time.time()
    
    # --- Load Raw Data ---
    if args.employee_file.endswith('.xlsx'):
        emp_df = pd.read_excel(args.employee_file)
    else:
        emp_df = pd.read_csv(args.employee_file)
        
    if args.request_file.endswith('.xlsx'):
        req_df = pd.read_excel(args.request_file)
    else:
        req_df = pd.read_csv(args.request_file)
    
    # --- Preprocess Data ---
    elapsed_pre, processed_df = preprocess_employee_data(emp_df, req_df)
    print(f"Preprocessing completed in {elapsed_pre:.2f} seconds.")
    
    # --- Prepare Data for Modeling ---
    # Assume the processed data includes a target column "BAJA_VOL". If not, generate a dummy target for demonstration.
    if 'BAJA_VOL' not in processed_df.columns:
        processed_df['BAJA_VOL'] = np.random.randint(0, 2, size=processed_df.shape[0])
    
    # Separate features and target
    X = processed_df.drop('BAJA_VOL', axis=1)
    y = processed_df['BAJA_VOL']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # --- Train Model ---
    best_model, best_params, best_cv_score = train_model(X_train, y_train, scoring=make_scorer(recall_score, pos_label=1))
    print("Best hyperparameters:", best_params)
    print("Best cross-validation recall:", best_cv_score)
    
    # --- Evaluate Model on Test Set ---
    y_pred = best_model.predict(X_test)
    print("Test Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # --- Predict on All Data (or New Data) ---
    preds_proba = best_model.predict_proba(X)[:, 1]
    threshold = 0.5  # Default threshold; can be adjusted as needed
    preds = (preds_proba >= threshold).astype(int)
    print(f"Predicted turnover percentage: {preds.mean()*100:.2f}%")
    
    # --- SHAP Analysis for Model Interpretability ---
    explainer = shap.Explainer(best_model)
    # Compute SHAP values for a sample (first 100 rows)
    shap_values = explainer(X.iloc[:100])
    # Generate and save summary plot
    shap.summary_plot(shap_values, X.iloc[:100], show=False)
    plt.savefig("shap_summary.png")
    plt.close()
    print("SHAP summary plot saved as 'shap_summary.png'.")
    
    # --- Save Predictions ---
    output_df = X.copy()
    output_df['Actual_Turnover'] = y
    output_df['Predicted_Probability'] = preds_proba
    output_df['Predicted_Turnover'] = preds
    output_df.to_csv("employee_turnover_predictions.csv", index=False)
    print("Predictions saved to 'employee_turnover_predictions.csv'.")
    
    total_elapsed = time.time() - start_time
    print(f"Total production pipeline completed in {total_elapsed/60:.2f} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production pipeline for employee turnover prediction.")
    parser.add_argument("--employee_file", type=str, required=True, help="Path to raw employee data file (Excel or CSV)")
    parser.add_argument("--request_file", type=str, required=True, help="Path to raw request data file (Excel or CSV)")
    args = parser.parse_args()
    main(args)
