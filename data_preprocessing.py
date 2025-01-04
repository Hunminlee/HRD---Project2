import pyreadstat
import numpy as np
import pandas as pd

def call_X_data(year, target):
    path = './SPSS/'
    file_path = path + f'{target}_{year}.sav'  # Original dataset model with different name for flexible data input

    dataframe, meta = pyreadstat.read_sav(file_path)

    return dataframe


def count_nan_and_extract_cols(data_series):
    cnt, lst = [], []
    for i in range(len(data_series)):
        if data_series[i] != 0:
            # print(data_series.index[i], data_series[i]) #Nan값 대부분 1000개넘음
            cnt.append(data_series[i])
            lst.append(data_series.index[i])

    return lst, cnt


def Check_String(df):
    # Identify columns containing string (object) data types
    string_columns = df.select_dtypes(include=['object']).columns

    # Check if there are any non-numeric values in these columns
    non_numeric_values = df[string_columns].applymap(type).eq(str).any(axis=1)

    if not string_columns.empty:
        print("Columns with potential string values:", string_columns.tolist())
        print(f"Number of rows with string values: {non_numeric_values.sum()}")
        # if non_numeric_values.sum() > 0:
        #    print("Sample rows with string values:")
        #    print(df[non_numeric_values])

    else:
        print("No string columns found in the DataFrame.")


def extract_features(var):
    # Example columns for feature names for each year
    X_df = var[var['분류'] == 'Feature_2']
    X_var_names = {2020: X_df['분류'].tolist(),
                   2021: X_df['Unnamed: 4'].tolist(),
                   2022: X_df['Unnamed: 5'].tolist(),
                   2023: X_df['Unnamed: 6'].tolist()}
    return X_var_names