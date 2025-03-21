""" This file contains methods to perform automated data cleaning using pandas / python. """

import pandas as pd

# Loading and exploring data
df = pd.read_csv('../../data/titanic_train.csv')

# start by performing a basic data quality check
def check_data_quality(df):
    # Store initial data quality metrics
    quality_report = {
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'total_rows': len(df),
        'memory_usage': df.memory_usage().sum() / 1024**2  # in MB
    }
    return quality_report

# converting string datas to datetime objects
# identifying and converting numeric strings to numbers
# ensuring categorical variables are properly encoded
def standardize_datatypes(df):
    for column in df.columns:
        # Try converting string dates to datetime
        if df[column].dtype == 'object':
            try:
                df[column] = pd.to_datetime(df[column])
                print(f"Converted {column} to datetime")
            except ValueError:
                # Try converting to numeric if datetime fails
                try:
                    df[column] = pd.to_numeric(df[column].str.replace('$', '').str.replace(',', ''))
                    print(f"Converted {column} to numeric")
                except:
                    pass
    return df

# handle missing values
# use median imputations for numeric columns
# apply mode imputation for categorical data
# maintain the statistical properties of the dataset while filling gaps
from sklearn.impute import SimpleImputer

def handle_missing_values(df):
    # Handle numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_columns) > 0:
        num_imputer = SimpleImputer(strategy='median')
        df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])

    # Handle categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

    return df

# detect and handle outliers
# calculate Interquartile range (IQR) for numeric columns
# identify values beyond 1.5 * IQR
# apply capping to extreme values rather than removing them
def remove_outliers(df):
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    outliers_removed = {}

    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers before removing
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]

        # Cap the values instead of removing them
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

        if outliers > 0:
            outliers_removed[column] = outliers

    return df, outliers_removed

# validate the results
def validate_cleaning(df, original_shape, cleaning_report):
    validation_results = {
        'rows_remaining': len(df),
        'missing_values_remaining': df.isnull().sum().sum(),
        'duplicates_remaining': df.duplicated().sum(),
        'data_loss_percentage': (1 - len(df) / original_shape[0]) * 100
    }

    # Add validation results to the cleaning report
    cleaning_report['validation'] = validation_results
    return cleaning_report

# put the whole thing together into a cleaning pipeline
def automated_cleaning_pipeline(df):
    # Store original shape for reporting
    original_shape = df.shape

    # Initialize cleaning report
    cleaning_report = {}

    # Execute each step and collect metrics
    cleaning_report['initial_quality'] = check_data_quality(df)

    df = standardize_datatypes(df)
    df = handle_missing_values(df)
    df, outliers = remove_outliers(df)
    cleaning_report['outliers_removed'] = outliers

    # Validate and finalize report
    cleaning_report = validate_cleaning(df, original_shape, cleaning_report)

    return df, cleaning_report

print(check_data_quality(df))