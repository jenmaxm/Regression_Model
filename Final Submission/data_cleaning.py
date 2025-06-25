import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Function to replace negative values with NaN
def replace_negative(x):
    if isinstance(x, (int, float)) and x < 0:
        return np.nan
    return x

# Function to clean data with advanced outlier handling and imputation
def clean_data(df):
    location_columns = df.columns[1:]  # All columns except the first one ("Date")
    
    # Iterate through each column (except the 'Date' column)
    for col in location_columns:
        # Step 1: Replace negative values with NaN
        df[col] = df[col].apply(replace_negative)

        # Step 2: Convert values to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Step 3: Handle outliers using Z-score (beyond a threshold)
        z_scores = zscore(df[col].dropna())  # Calculate Z-scores
        outlier_indices = np.abs(z_scores) > 4  # Outliers if Z-score > 4
        
        # Re-align outlier_indices with the DataFrame's index
        outlier_indices = pd.Series(outlier_indices, index=df[col].dropna().index)

        # Replace outliers with NaN
        df.loc[outlier_indices.index, col] = df.loc[outlier_indices.index, col].where(~outlier_indices, np.nan)
        
        # Step 4: Handle missing data
        # Use rolling median as a more robust method to fill missing data
        df[col] = df[col].fillna(df[col].rolling(window=3, min_periods=1).median())  # Median imputation
        
        # Step 5: Handle residual missing data by interpolation
        df[col] = df[col].interpolate(method='linear', limit_direction='both')  # Linear interpolation

        # Step 6: Handle extreme outliers using IQR method (optional step)
        Q1 = np.percentile(df[col].dropna(), 25)
        Q3 = np.percentile(df[col].dropna(), 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan
        
        # Step 7: Forward fill and backward fill for remaining NaNs
        df[col] = df[col].ffill().bfill()

    return df

# Function to add lag features to the data
def add_lags(df, lag_days=1):
    for col in df.columns[1:]:  
        for lag in range(1, lag_days + 1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

def load_and_clean_data(file_path, lag_days=4, shift_target=True):
    # Load the dataset
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(xls, sheet_name="1993-96", skiprows=1)

    # Rename columns
    df.columns = ["Date", "Crakehill", "Skip Bridge", "Westwick", "Skelton", 
                  "Arkengarthdale", "East Cowton", "Malham Tarn", "Snaizeholme", 
                  "Extra1", "Extra2", "Notes"]

    # Drop unnecessary columns
    df = df.drop(columns=["Extra1", "Extra2", "Notes"], errors='ignore')

    # Convert 'Date' to datetime format
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

    # Drop rows where all values (except Date) are NaN
    df = df.dropna(subset=df.columns[1:], how='all')

    # Clean data
    df_cleaned = clean_data(df)

    # Shift Skelton only if required
    if shift_target:
        df_cleaned['Skelton'] = df_cleaned['Skelton'].shift(-1)

    # Add lag features
    df_cleaned = add_lags(df_cleaned, lag_days=lag_days)

    # Drop NaN rows from lagging
    df_cleaned = df_cleaned.dropna()

    return df_cleaned


