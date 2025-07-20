import pandas as pd
import numpy as np

def load_data(cfg, start_datetime=None, end_datetime=None):
    df = pd.read_csv(cfg.data_path, sep=',')
    print("Columns:", df.columns.tolist())
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values(by='timestamp')
    
    # Filter data based on start_datetime and end_datetime if provided
    if start_datetime is not None:
        df = df[df['timestamp'] >= start_datetime]
    if end_datetime is not None:
        df = df[df['timestamp'] <= end_datetime]
    
    y_all = np.log(df['price'].astype(float).values)
    X = np.arange(len(y_all))  # Re-index X after filtering
    
    return X, y_all