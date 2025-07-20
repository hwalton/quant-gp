import pandas as pd
import os

# File paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEEKLY_DATA = os.path.join(PROJECT_ROOT, 'a_data', 'BitcoinHistory.csv')
MINUTE_DATA = os.path.join(PROJECT_ROOT, 'a_data', 'btcusd_1-min_data.csv')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'a_data', 'bitcoin_combined_weekly_data.csv')

def process_weekly_data(file_path):
    """Process daily Bitcoin data and resample to weekly (Sunday close)"""
    df = pd.read_csv(file_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean price columns first
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean volume column (remove 'K', 'M', 'B' suffixes)
    df['Vol.'] = df['Vol.'].astype(str).str.replace('-', '0')
    
    def convert_volume(vol_str):
        if pd.isna(vol_str) or vol_str == '0':
            return 0
        vol_str = str(vol_str).upper()
        if vol_str.endswith('K'):
            return float(vol_str[:-1]) * 1000
        elif vol_str.endswith('M'):
            return float(vol_str[:-1]) * 1000000
        elif vol_str.endswith('B'):
            return float(vol_str[:-1]) * 1000000000
        else:
            return float(vol_str)
    
    df['Vol.'] = df['Vol.'].apply(convert_volume)
    
    # Clean change_pct column
    df['Change %'] = df['Change %'].astype(str).str.replace('%', '').replace('-', '0')
    df['Change %'] = pd.to_numeric(df['Change %'], errors='coerce').fillna(0)
    
    # Set Date as index for resampling
    df = df.set_index('Date')
    df = df.sort_index()
    
    # Resample to weekly data (Sunday close)
    # 'W' means weekly ending on Sunday
    weekly_df = df.resample('W').agg({
        'Price': 'last',      # Close price (last price of the week)
        'Open': 'first',      # Open price (first price of the week)
        'High': 'max',        # Highest price of the week
        'Low': 'min',         # Lowest price of the week
        'Vol.': 'sum',        # Total volume for the week
        'Change %': lambda x: ((1 + x/100).prod() - 1) * 100  # Compound weekly change
    }).reset_index()
    
    # Convert Date back to timestamp
    weekly_df['timestamp'] = weekly_df['Date'].astype('int64') // 10**9
    
    # Rename columns to match expected format
    weekly_df = weekly_df.rename(columns={
        'Price': 'price',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Vol.': 'volume',
        'Change %': 'change_pct'
    })
    
    # Select final columns
    weekly_df = weekly_df[['timestamp', 'price', 'open', 'high', 'low', 'volume', 'change_pct']]
    
    # Sort by timestamp
    weekly_df = weekly_df.sort_values('timestamp')
    
    return weekly_df

def process_minute_data(file_path, last_timestamp):
    """Process minute-level Bitcoin data"""
    # First, let's check what columns exist
    df = pd.read_csv(file_path)
    print(f"Columns in minute data: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head())
    
    # Common column name variations for timestamp
    timestamp_cols = ['timestamp', 'Timestamp', 'time', 'Time', 'date', 'Date', 'datetime', 'DateTime']
    timestamp_col = None
    
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print("Available columns:", df.columns.tolist())
        raise ValueError("No timestamp column found. Please check your data file.")
    
    print(f"Using timestamp column: {timestamp_col}")
    
    # Convert timestamp to unix timestamp if it's not already
    if df[timestamp_col].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df[timestamp_col]).astype('int64') // 10**9
    else:
        df['timestamp'] = df[timestamp_col]
    
    # Filter for data after the last weekly timestamp
    df = df[df['timestamp'] > last_timestamp]
    
    if df.empty:
        print("No new minute data found after the last weekly timestamp")
        return pd.DataFrame()
    
    # Map common column names to our standard names
    column_mapping = {
        'close': 'price',
        'Close': 'price',
        'open': 'open',
        'Open': 'open',
        'high': 'high',
        'High': 'high',
        'low': 'low',
        'Low': 'low',
        'volume': 'volume',
        'Volume': 'volume',
        'vol': 'volume',
        'Vol': 'volume'
    }
    
    # Rename columns based on mapping
    df = df.rename(columns=column_mapping)
    
    # Ensure we have the required columns
    required_cols = ['timestamp', 'price', 'open', 'high', 'low', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        # Fill missing columns with price if available
        if 'price' in df.columns:
            for col in ['open', 'high', 'low']:
                if col in missing_cols:
                    df[col] = df['price']
        if 'volume' in missing_cols:
            df['volume'] = 0
    
    # Add change_pct column
    df['change_pct'] = df['price'].pct_change() * 100
    df['change_pct'] = df['change_pct'].fillna(0)
    
    # Select only the columns we need
    df = df[['timestamp', 'price', 'open', 'high', 'low', 'volume', 'change_pct']]
    
    # Convert to weekly data by resampling (Sunday close)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('datetime')
    
    weekly_df = df.resample('W').agg({
        'price': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'volume': 'sum',
        'change_pct': lambda x: ((1 + x/100).prod() - 1) * 100  # Compound change
    }).reset_index()
    
    weekly_df['timestamp'] = weekly_df['datetime'].astype('int64') // 10**9
    weekly_df = weekly_df.drop('datetime', axis=1)
    
    return weekly_df

def main():
    print("Starting data processing...")
    
    # Process weekly data
    print(f"Processing {WEEKLY_DATA}...")
    weekly_df = process_weekly_data(WEEKLY_DATA)
    print(f"Processed {len(weekly_df)} weekly data points from {WEEKLY_DATA}")
    
    # Get the last timestamp from weekly data
    last_timestamp = weekly_df['timestamp'].max()
    print(f"Last timestamp from {WEEKLY_DATA}: {last_timestamp}")
    
    # Process minute data for recent data
    print(f"Processing {MINUTE_DATA} for recent data...")
    try:
        minute_df = process_minute_data(MINUTE_DATA, last_timestamp)
        
        if not minute_df.empty:
            print(f"Processed {len(minute_df)} weekly data points from minute data")
            
            # Combine the data
            combined_df = pd.concat([weekly_df, minute_df], ignore_index=True)
            combined_df = combined_df.sort_values('timestamp')
            
            # Remove duplicates based on timestamp
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
            
            print(f"Combined dataset has {len(combined_df)} weekly data points")
        else:
            combined_df = weekly_df
            print("Using only weekly data as no new minute data was found")
            
    except Exception as e:
        print(f"Error processing minute data: {e}")
        print("Using only weekly data")
        combined_df = weekly_df
    
    # Save the combined data
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Combined data saved to {OUTPUT_FILE}")
    
    # Display summary
    print("\nData Summary:")
    print(f"Date range: {pd.to_datetime(combined_df['timestamp'].min(), unit='s')} to {pd.to_datetime(combined_df['timestamp'].max(), unit='s')}")
    print(f"Total data points: {len(combined_df)}")
    print("\nFirst few rows:")
    print(combined_df.head())
    print("\nLast few rows:")
    print(combined_df.tail())

if __name__ == "__main__":
    main()