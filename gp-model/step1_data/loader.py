"""
Data loading and management for the QuantGP pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from config import GPModelConfig

def load_btc_data(config: GPModelConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Bitcoin price data and return log prices.
    
    Returns:
        tuple: (X, y) where X is time indices and y is log prices
    """
    # Try to load from new data directory first, then fallback to old location
    data_path = config.data_dir / config.btc_data_file
    if not data_path.exists():
        # Fallback to old location
        old_data_path = config.base_dir.parent / "0-data" / config.btc_data_file
        if old_data_path.exists():
            data_path = old_data_path
        else:
            raise FileNotFoundError(f"Bitcoin data file not found at {data_path} or {old_data_path}")
    
    df = pd.read_csv(data_path, sep=';')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    
    # Convert to log prices
    y = np.log(df['close'].astype(float).values)
    X = np.arange(len(y))
    
    return X, y

def copy_data_to_local(config: GPModelConfig) -> None:
    """Copy data from old location to new gp-model/data directory."""
    old_data_path = config.base_dir.parent / "0-data" / config.btc_data_file
    new_data_path = config.data_dir / config.btc_data_file
    
    if old_data_path.exists() and not new_data_path.exists():
        import shutil
        shutil.copy2(old_data_path, new_data_path)
        print(f"Copied data from {old_data_path} to {new_data_path}")

if __name__ == "__main__":
    # Test data loading
    config = GPModelConfig()
    copy_data_to_local(config)
    X, y = load_btc_data(config)
    print(f"Loaded {len(X)} data points")
    print(f"Log price range: {y.min():.3f} to {y.max():.3f}")
