# EmployeeTurnoverPredictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A predictive modeling pipeline for forecasting employee voluntary turnover using advanced machine learning techniques and model interpretability tools.

## Overview

This repository contains a complete pipeline to preprocess employee and request data, train predictive models for voluntary employee turnover, and generate interpretable explanations for the predictions. The project is built using Python and leverages popular libraries such as Pandas, NumPy, Scikit-learn, XGBoost, and SHAP.

## Repository Structure

- **employee_preprocess.py**  
  Contains all functions and routines to clean and preprocess raw employee data as well as request data. Key tasks include:
  - Filtering and cleaning data (e.g., removing outliers based on age).
  - Converting date fields and handling missing values.
  - Feature engineering (e.g., calculating time since last assignment, discretizing continuous variables, generating "under-salary" flags such as SIER, SIPET, SIPP, etc.).
  - Encoding categorical variables and scaling features.

- **employee_production.py**  
  Implements the production pipeline which:
  - Loads employee and request data (from files or databases).
  - Applies the pre-processing routine from `employee_preprocess.py`.
  - Splits data into training and test sets.
  - Performs hyperparameter tuning (using GridSearchCV) for an XGBoost classifier.
  - Trains the final model and calibrates predictions.
  - Uses SHAP to provide local interpretability for individual predictions.
  - Exports prediction results (and optionally writes them to a database).

## Data and Variables Used

The pipeline uses a variety of employee variables to build the predictive model. Key variables include:

- **Demographic & Personal:**
  - `age`: Employee age.
  - `education_level`: The highest education attained (engineer, FP, high school, etc.).

- **Contract and Tenure:**
  - `contract_type`: Indicates if the contract is permanent or temporary.
  - `months_of_service`: Number of months the employee has worked.
  - `months_since_last_review`: Time since the last salary or performance review.
  - `TSLA` (Time Since Last Assignment): Number of months since the employee’s last assignment.

- **Work and Performance:**
  - `hours_training`, `hours_absence`, `hours_service`: Metrics related to training, absence, and service hours.
  - `satisfaction_job`, `satisfaction_peers`, `satisfaction_company`: Survey scores indicating satisfaction levels (after scale inversion so that higher values represent higher satisfaction).
  - `salary`: The current salary of the employee.
  - `benchmark_salary`: Salary benchmarks computed from similar profiles or recent hires.
  - `salary_improvement_percentage`: The percentage increase in the last salary revision.
  - Engineered flags such as:
    - **SIER:** Flag indicating if the employee’s salary is below the median of recently hired employees in the same profile.
    - **SIPET:** Flag based on salary desired in recent talent requests.
    - **SIPP:** Flag based on salary compared to the median salary of the professional profile.
    - **SIPC:** Flag based on salary compared to the average salary of the employee’s position within the same branch/department.
    - **SICT:** Flag based on salary relative to the work center’s (office) average salary.
  - Additional features from client and service data (e.g., number of clients in the last period, indicators of high client turnover).

- **Engineered Aggregates:**
  - Categorical groupings (e.g., discretization of continuous variables into quartiles).
  - Aggregated features such as the total count of “under-salary” flags and “high turnover” indicators.

## How to Run the Pipeline

1. **Installation:**  
   Ensure you have Python 3 installed. Install the required libraries via pip:
   ```bash
   pip install pandas numpy scikit-learn xgboost shap mysql-connector-python sqlalchemy
   ```

2. **Configuration:**  
   - Update the connection parameters (host, database, user, password) in the production scripts if you plan to load data from a database.
   - Alternatively, adjust the file paths to point to your local CSV or Excel files containing `employee_data` and `request_data`.

3. **Preprocessing:**  
   Run the pre-processing script to generate the cleaned data:
   ```bash
   python employee_preprocess.py --employee_file path/to/employee_data.xlsx --request_file path/to/request_data.csv
   ```

4. **Training and Prediction:**  
   Run the production pipeline:
   ```bash
   python employee_production.py --employee_file path/to/employee_data.xlsx --request_file path/to/request_data.csv
   ```
   *(If you use a database connection, ensure the script is updated with your connection parameters.)*

5. **Results:**  
   The script will output:
   - A trained model with calibrated probabilities.
   - Prediction results with predicted turnover probabilities.
   - Local explanations for each employee prediction using SHAP (with a summary plot saved as `shap_summary.png`).
   - A CSV file (`employee_turnover_predictions.csv`) containing the predictions and related data.

## Replication & Customization

- The pipeline is fully modular. You can:
  - Swap the modeling algorithm by modifying the hyperparameter grid and model instantiation in `employee_production.py`.
  - Extend or modify the feature engineering in `employee_preprocess.py` by editing or adding new functions.
  - Adjust the thresholding logic to match the desired proportion of predicted turnovers.

## Contributing

Contributions, suggestions, and improvements are welcome. Please open an issue or submit a pull request if you have any ideas or corrections.

## License

This project is licensed under the MIT License.
```
