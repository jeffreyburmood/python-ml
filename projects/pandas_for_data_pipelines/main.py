"""
This program will explore using data pipelines in pandas

    To create an end-to-end data science pipeline, we first have to convert the above code into a proper format using Python functions.

We will create Python functions for:

Loading the data: It requires a directory of CSV files.
Cleaning the data: It requires raw DataFrame and returns the cleaned DataFrame.
Convert column types: It requires a clean DataFrame and data types and returns the DataFrame with the correct data types.
Data analysis: It requires a DataFrame from the previous step and returns the modified DataFrame with two columns.
Data visualization: It requires a modified DataFrame and visualization type to generate visualization.
"""

import pandas as pd

def main():
    path = '../../data/Online_Sales_Data.csv'
    df = (pd.DataFrame()
          .pipe(lambda x: load_data(path))
          .pipe(data_cleaning)
          .pipe(convert_dtypes, {'Product Category': 'str', 'Product Name': 'str'})
          .pipe(data_analysis)
          .pipe(data_visualization, 'line')
          )

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop_duplicates()
    data = data.dropna()
    data = data.reset_index(drop=True)
    return data

def convert_dtypes(data: pd.DataFrame, types_dict=None):
    data = data.astype(dtype=types_dict)
    # convert the date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def data_analysis(data: pd.DataFrame) -> pd.DataFrame:
    data['month'] = data['Date'].dt.month
    new_df = data.groupby('month')['Units Sold'].mean()
    return new_df

def data_visualization(new_df: pd.DataFrame, vis_type: str='bar'):
    new_df.plot(kind=vis_type, figsize=(10, 5), title='Average Units Sold by Month')
    return new_df

if __name__ == "__main__":
    main()