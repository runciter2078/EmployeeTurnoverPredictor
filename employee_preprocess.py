#!/usr/bin/env python3
"""
employee_preprocess.py

Version: 1.0
Description:
    This module preprocesses raw employee and request data for the employee turnover prediction model.
    It performs cleaning, filtering, feature engineering, encoding and scaling of the data.
    
Usage:
    python employee_preprocess.py --employee_file path/to/employee_data.xlsx --request_file path/to/request_data.csv

The script outputs a processed CSV file ("processed_employee_data.csv") and prints the elapsed processing time.
"""

import pandas as pd
import numpy as np
import re
import time
from datetime import date, datetime, timedelta
from dateutil import relativedelta
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
np.random.seed(1)

def fill_recent_salary(employee, salary_tables):
    """
    Look up a benchmark salary using a hierarchy of salary tables.

    Args:
        employee (pd.Series): A row from the employee DataFrame containing 'community', 'position' and 'level'.
        salary_tables (dict): A dictionary with keys 'cpn', 'cp' and 'c' that contain pivot tables with salary medians.

    Returns:
        float: The benchmark salary.
    """
    community = employee['community']
    position = employee['position']
    level = employee['level']
    try:
        return salary_tables['cpn'].loc[(community, position, level)][0]
    except Exception:
        try:
            return salary_tables['cp'].loc[(community, position)][0]
        except Exception:
            try:
                return salary_tables['c'].loc[(community)][0]
            except Exception:
                # Fallback: use the median from the most detailed table
                return salary_tables['cpn'].median()[0]

def time_since_in_months(date_start, date_end):
    """
    Calculates the number of months between two dates.
    If the end date is in the future relative to today, uses today's date.

    Args:
        date_start (datetime.date): Starting date.
        date_end (datetime.date): Ending date.

    Returns:
        int: Number of months difference, or -1 on error.
    """
    try:
        today = datetime.combine(date.today(), datetime.min.time())
        dS = datetime.combine(date_start, datetime.min.time())
        dE = datetime.combine(date_end, datetime.min.time())
        if dE > today:
            diff = relativedelta.relativedelta(today, dS)
        else:
            diff = relativedelta.relativedelta(dE, dS)
        return diff.years * 12 + diff.months
    except Exception:
        return -1

def mask_work_center(center):
    """
    Maps the work center string to a generic region.
    
    Args:
        center (str): Original work center string.
        
    Returns:
        str: Mapped region.
    """
    words = center.lower().split()
    if 'rozas' in words or 'retama' in words:
        return 'Madrid'
    elif 'barcelona' in words:
        return 'Barcelona'
    elif 'jerez' in words or 'sevilla' in words or 'huelva' in words:
        return 'Andalusia'
    else:
        return 'Other'

def preprocess_employee_data(employee_df, request_df):
    """
    Preprocess employee and request data.

    Args:
        employee_df (pd.DataFrame): Raw employee data.
        request_df (pd.DataFrame): Raw request data.

    Returns:
        tuple: (elapsed_time in seconds, processed employee DataFrame with engineered features)
    """
    start_time = time.time()
    
    # --- Shuffle and Filter ---
    employee_df = employee_df.sample(frac=1, random_state=1)
    # Filter by technical community and exclude specific contract type
    employee_df = employee_df[employee_df['technical_community'] == 1]
    employee_df = employee_df[employee_df['contract_type'] != 'Mercantil']
    
    # --- Date Conversions ---
    # Convert date columns to datetime
    date_cols = ['start_date', 'end_date', 'last_assignment_date']
    for col in date_cols:
        employee_df[col] = pd.to_datetime(employee_df[col], dayfirst=True, errors='coerce')
    
    # --- Age Filtering ---
    # Remove employees with age <= 16 or >= 75
    employee_df = employee_df[(employee_df['age'] > 16) & (employee_df['age'] < 75)]
    
    # --- Fill Missing Numeric Values ---
    numeric_cols = ['hours_training', 'hours_absence', 'hours_service']
    for col in numeric_cols:
        employee_df[col] = employee_df[col].fillna(0)
    
    # --- Process Request Data ---
    req_date_cols = ['request_date', 'change_date', 'closure_date', 'need_date']
    for col in req_date_cols:
        request_df[col] = pd.to_datetime(request_df[col], errors='coerce').dt.date
    # Filter requests from last 365 days
    days_model = 365
    cutoff_date = date.today() - timedelta(days=days_model)
    request_df = request_df[request_df['request_date'] >= cutoff_date]
    
    # --- Create Salary Pivot Tables from Request Data ---
    salary_table_cpn = request_df.pivot_table('desired_salary',
                                              index=['community', 'position', 'level'],
                                              aggfunc=np.median)
    salary_table_cp = request_df.pivot_table('desired_salary',
                                             index=['community', 'position'],
                                             aggfunc=np.median)
    salary_table_c = request_df.pivot_table('desired_salary',
                                            index=['community'],
                                            aggfunc=np.median)
    salary_tables = {'cpn': salary_table_cpn, 'cp': salary_table_cp, 'c': salary_table_c}
    
    # --- Feature Engineering ---
    # TSLA: Time Since Last Assignment (in months)
    employee_df['TSLA'] = employee_df['last_assignment_date'].apply(
        lambda d: time_since_in_months(d.date(), date.today()) if pd.notnull(d) else -1)
    employee_df['TSLA'] = employee_df['TSLA'].clip(upper=240)
    
    # Benchmark salary: based on similar profiles from request data
    employee_df['benchmark_salary'] = employee_df.apply(
        lambda row: fill_recent_salary(row, salary_tables), axis=1)
    
    # Flag SIER: salary below benchmark
    employee_df['SIER'] = np.where(employee_df['salary'] < employee_df['benchmark_salary'], 1, 0)
    
    # Example: Map work center to region
    if 'work_center' in employee_df.columns:
        employee_df['region'] = employee_df['work_center'].apply(mask_work_center)
    
    # Additional feature engineering can be added here...
    
    # --- Encoding and Scaling ---
    # One-hot encode categorical variables (object dtype)
    cat_cols = employee_df.select_dtypes(include=['object']).columns.tolist()
    employee_encoded = pd.get_dummies(employee_df, columns=cat_cols, drop_first=True)
    
    # Scale all numeric features
    scaler = MinMaxScaler()
    employee_encoded[employee_encoded.columns] = scaler.fit_transform(employee_encoded[employee_encoded.columns])
    
    elapsed = time.time() - start_time
    return elapsed, employee_encoded

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess employee and request data for turnover prediction.")
    parser.add_argument("--employee_file", type=str, required=True, help="Path to employee data file (Excel or CSV)")
    parser.add_argument("--request_file", type=str, required=True, help="Path to request data file (Excel or CSV)")
    args = parser.parse_args()
    
    # Load employee data
    if args.employee_file.endswith('.xlsx'):
        emp_df = pd.read_excel(args.employee_file)
    else:
        emp_df = pd.read_csv(args.employee_file)
    
    # Load request data
    if args.request_file.endswith('.xlsx'):
        req_df = pd.read_excel(args.request_file)
    else:
        req_df = pd.read_csv(args.request_file)
    
    elapsed, processed_df = preprocess_employee_data(emp_df, req_df)
    print(f"Preprocessing completed in {elapsed:.2f} seconds.")
    processed_df.to_csv("processed_employee_data.csv", index=False)
