import os
import json
import pandas as pd
import numpy as np

def clean_numeric(value):
    """
    Convert a value to float, return NaN if conversion fails.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def compute_rsi(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) indicator.
    """
    delta = data.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))

    return pd.Series(rsi, index=data.index)

def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculate the MACD and its signal line.
    """
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def compute_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands (upper and lower) using moving average and standard deviation.
    """
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    return upper_band, lower_band

def compute_atr(df, window=14):
    """
    Calculate the Average True Range (ATR) indicator.
    """
    tr = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift()),
        abs(df["Low"] - df["Close"].shift())
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr

def compute_obv(df):
    """
    Calculate the On-Balance Volume (OBV) indicator.
    """
    obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    return obv

def process_json_file(file_path, output_folder="processed_stock_data_1"):
    """
    Process a single stock JSON file: clean, extract, and add technical indicators.
    """
    with open(file_path, "r") as file:
        data = json.load(file)

    df = pd.DataFrame(data)

    # Parse and clean date
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Clean and convert numeric fields
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(str).str.replace(",", "").apply(clean_numeric)

    df = df.dropna().reset_index(drop=True)
    df.set_index("Date", inplace=True)

    # Compute technical indicators
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["RSI_14"] = compute_rsi(df["Close"], 14)
    df["MACD"], df["Signal_Line"] = compute_macd(df["Close"])
    df["Upper_Band"], df["Lower_Band"] = compute_bollinger_bands(df["Close"])
    df["ATR_14"] = compute_atr(df)
    df["OBV"] = compute_obv(df)

    # Handle missing values (forward and backward fill)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # Smooth OBV if variance is too high
    if df["OBV"].std() > df["OBV"].mean() * 10:
        df["OBV"] = df["OBV"].rolling(window=5, min_periods=1).mean()

    # Save processed data
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, os.path.basename(file_path))
    df.reset_index().to_json(output_file, orient="records", date_format="iso")
    print(f"Processed: {file_path} → {output_file}")

def process_all_json_files(input_folder="stock_data", output_folder="processed_stock_data_1"):
    """
    Iterate through all JSON files in a folder and process them.
    """
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_folder, file_name)
            process_json_file(file_path, output_folder)

process_all_json_files()
