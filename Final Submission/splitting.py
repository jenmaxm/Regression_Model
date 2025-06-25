import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_cleaning import load_and_clean_data

def split_and_scale_data(file_path):
    # Load and clean the data
    df = load_and_clean_data(file_path)    
    # Convert 'Date' to datetime if necessary
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Feature engineering
    df['CWSB_Rolling_Mean_5'] = (df['Crakehill'] + df['Westwick'] + df['Skip Bridge']).rolling(window=5).mean()
    df['Date_in_days'] = (df['Date'] - df['Date'].min()).dt.days
    df['Flow_Diff_Crakehill_Skip_Bridge'] = df['Crakehill'] - df['Skip Bridge']
    df['Flow_Diff_Crakehill_Skip_Bridge_Rolling_Mean_3'] = df['Flow_Diff_Crakehill_Skip_Bridge'].rolling(window=3).mean()
    df['Skelton_Lag1'] = df['Skelton'].shift(1)
    df["RainStations"] = df["Arkengarthdale"] + df["East Cowton"] + df["Snaizeholme"]
    df['Westwick_Rolling_Mean_7'] = df['Westwick'].rolling(window=7).mean()

    df['Crakehill_Pct_Change'] = df['Crakehill'].pct_change()
    df['Westwick_Pct_Change'] = df['Westwick'].pct_change()
    df['SkipBridge_Pct_Change'] = df['Skip Bridge'].pct_change()
    
    df['Crakehill_SkipBridge_Interaction'] = df['Crakehill_Pct_Change'] * df['SkipBridge_Pct_Change']
    df['Westwick_SkipBridge_Interaction'] = df['Westwick_Pct_Change'] * df['SkipBridge_Pct_Change']
    
    # API_7 (Exponential moving average for Combined Stations)
    df['Combined_Stations'] = df['Arkengarthdale'] + df['East Cowton'] + df['Snaizeholme']
    df['API_7'] = df['Combined_Stations'].ewm(alpha=1/7, adjust=False).mean()
    
    # Flow_Propagation_Ratio (Skelton to Crakehill ratio)
    df['Flow_Propagation_Ratio'] = df['Skelton'] / (df['Crakehill'] + 1e-5)  # Prevent division by zero
    
    # Drop NaN values
    df.dropna(subset=['CWSB_Rolling_Mean_5', 'Date_in_days', 'Flow_Diff_Crakehill_Skip_Bridge_Rolling_Mean_3', 
                      'Skelton_Lag1', 'RainStations', 'Westwick_Rolling_Mean_7', 'API_7', 'Flow_Propagation_Ratio',
                      'Crakehill_SkipBridge_Interaction'
                      ,'Westwick_SkipBridge_Interaction'], inplace=True)
    
    # Extract year for splitting
    df["Year"] = df["Date"].dt.year
    years = sorted(df["Year"].unique())
    train_years = years[:len(years) // 2]
    valid_years = years[len(years) // 2 : len(years) * 3 // 4]
    test_years = years[len(years) * 3 // 4 :]
    
    # Split data
    train_df = df[df["Year"].isin(train_years)].drop(columns=["Year", "Date"])
    valid_df = df[df["Year"].isin(valid_years)].drop(columns=["Year", "Date"])
    test_df = df[df["Year"].isin(test_years)].drop(columns=["Year", "Date"])
    
    
    # Define features and target
    features = [
        "CWSB_Rolling_Mean_5", "Date_in_days", 
        "Flow_Diff_Crakehill_Skip_Bridge_Rolling_Mean_3",
        "Skelton_Lag1", "RainStations", "Westwick_Rolling_Mean_7",
        "API_7", "Flow_Propagation_Ratio"
    ]
    
    df["Target"] = df['Skelton'].shift(-1)  # Predict the next time step for Skelton
    target = "Skelton"
    
    # Scale the data
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[features])
    valid_features = scaler.transform(valid_df[features])
    test_features = scaler.transform(test_df[features])
    
    
    # Return scaled data
    return (
        pd.DataFrame(train_features, columns=features, index=train_df.index),
        pd.DataFrame(valid_features, columns=features, index=valid_df.index),
        pd.DataFrame(test_features, columns=features, index=test_df.index),
        train_df[target].reset_index(drop=True),
        valid_df[target].reset_index(drop=True),
        test_df[target].reset_index(drop=True)
    )
