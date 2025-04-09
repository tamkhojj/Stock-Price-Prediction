import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def process_stock_file(file_path, output_folder):
    """
    Load a single stock JSON file, clean and normalize the numeric columns,
    and save the processed data to the specified output folder.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()
        if raw.strip().startswith("{") and "},{" in raw:
            raw = "[" + raw + "]"
        data = json.loads(raw)

    df = pd.DataFrame(data)

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Convert numeric columns and handle missing values
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0 if df[col].isna().all() else df[col].mean())

    # Normalize numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    df.to_json(output_path, orient="records", date_format="iso", force_ascii=False)

def process_all_stock_files(input_folder, output_folder):
    """
    Iterate over all JSON files in the input folder,
    process and save each to the output folder.
    """
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            process_stock_file(file_path, output_folder)

if __name__ == "__main__":
    input_folder = "processed_stock_data_1"
    output_folder = "processed_stock_data_scaled"
    process_all_stock_files(input_folder, output_folder)
