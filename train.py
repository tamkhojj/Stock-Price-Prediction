import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
from joblib import dump
from tqdm import tqdm

MODEL_NAMES = ['RandomForest', 'XGBoost', 'LinearRegression', 'ARIMA', 'LSTM']
for model in MODEL_NAMES:
    os.makedirs(f"trained_models/{model}", exist_ok=True)
    os.makedirs(f"charts/{model}", exist_ok=True)

def load_json_file(filepath):
    with open(filepath, 'r') as f:
        return pd.DataFrame(json.load(f))

def evaluate_model(y_true, y_pred, prefix=""):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{prefix} MAE: {mae:.2f}")
    print(f"{prefix} MSE: {mse:.2f}")
    print(f"{prefix} RMSE: {rmse:.2f}")
    print(f"{prefix} R²: {r2:.2f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def plot_predictions(df, actual, train_pred, test_pred, title, output_path):
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, actual, label="Actual Prices", color='green')
    plt.plot(df.index[:len(train_pred)], train_pred, label="Train Predictions", color='orange')
    plt.plot(df.index[len(train_pred):len(train_pred)+len(test_pred)], test_pred, label="Test Predictions", color='teal')
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def train_models_on_file(filepath):
    df = load_json_file(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.dropna()

    if df.shape[0] < 100:
        return

    features = df.columns.difference(['Close'])
    X = df[features].values
    y = df['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model_scores = {}
    stock_name = os.path.basename(filepath).replace('.json', '')

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    dump(rf, f"trained_models/RandomForest/{stock_name}.joblib")
    y_rf_train = rf.predict(X_train)
    y_rf_test = rf.predict(X_test)
    model_scores["RandomForest"] = evaluate_model(y_test, y_rf_test, "Random Forest Test")
    plot_predictions(df, y, y_rf_train, y_rf_test, f"{stock_name} - Random Forest", f"charts/RandomForest/{stock_name}.png")

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    dump(xgb, f"trained_models/XGBoost/{stock_name}.joblib")
    y_xgb_train = xgb.predict(X_train)
    y_xgb_test = xgb.predict(X_test)
    model_scores["XGBoost"] = evaluate_model(y_test, y_xgb_test, "XGBoost Test")
    plot_predictions(df, y, y_xgb_train, y_xgb_test, f"{stock_name} - XGBoost", f"charts/XGBoost/{stock_name}.png")

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    dump(lr, f"trained_models/LinearRegression/{stock_name}.joblib")
    y_lr_train = lr.predict(X_train)
    y_lr_test = lr.predict(X_test)
    model_scores["LinearRegression"] = evaluate_model(y_test, y_lr_test, "Linear Regression Test")
    plot_predictions(df, y, y_lr_train, y_lr_test, f"{stock_name} - Linear Regression", f"charts/LinearRegression/{stock_name}.png")

    # ARIMA
    try:
        arima = ARIMA(y, order=(5,1,0))
        arima_model = arima.fit()
        y_arima_pred = arima_model.predict(start=len(y_train), end=len(y)-1, typ='levels')
        model_scores["ARIMA"] = evaluate_model(y_test, y_arima_pred, "ARIMA Test")
        plot_predictions(df, y, y[:len(y_train)], y_arima_pred, f"{stock_name} - ARIMA", f"charts/ARIMA/{stock_name}.png")
    except Exception as e:
        model_scores["ARIMA"] = f"ARIMA Failed: {e}"

    # LSTM
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Close']])
        SEQ_LEN = 10

        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i+seq_len])
                y.append(data[i+seq_len])
            return np.array(X), np.array(y)

        X_lstm, y_lstm = create_sequences(scaled_data, SEQ_LEN)
        X_train_lstm, X_test_lstm = X_lstm[:int(len(X_lstm)*0.8)], X_lstm[int(len(X_lstm)*0.8):]
        y_train_lstm, y_test_lstm = y_lstm[:int(len(X_lstm)*0.8)], y_lstm[int(len(X_lstm)*0.8):]

        model_lstm = Sequential()
        model_lstm.add(LSTM(64, return_sequences=False, input_shape=(X_train_lstm.shape[1], 1)))
        model_lstm.add(Dense(1))
        model_lstm.compile(optimizer='adam', loss='mse')

        es = EarlyStopping(patience=5, restore_best_weights=True)
        model_lstm.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=16, validation_split=0.1, callbacks=[es], verbose=0)
        y_pred_lstm = model_lstm.predict(X_test_lstm)
        y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm)
        y_test_lstm_inv = scaler.inverse_transform(y_test_lstm)

        model_scores["LSTM"] = evaluate_model(y_test_lstm_inv, y_pred_lstm_inv, "LSTM Test")
        model_lstm.save(f"trained_models/LSTM/{stock_name}.keras")

        full_actual = scaler.inverse_transform(scaled_data[SEQ_LEN:])
        full_pred = np.concatenate([
            scaler.inverse_transform(model_lstm.predict(X_lstm[:len(X_train_lstm)])),
            y_pred_lstm_inv
        ])
        plot_predictions(df.iloc[SEQ_LEN:], full_actual.flatten(), full_pred[:len(X_train_lstm)], full_pred[len(X_train_lstm):], f"{stock_name} - LSTM", f"charts/LSTM/{stock_name}.png")

    except Exception as e:
        model_scores["LSTM"] = f"LSTM Failed: {e}"

    return model_scores

folder = "/kaggle/input/processed-stock-dataset/processed_stock_data_scaled"
all_scores = {}
for filename in tqdm(os.listdir(folder)):
    if filename.endswith(".json"):
        file_path = os.path.join(folder, filename)
        result = train_models_on_file(file_path)
        all_scores[filename] = result
     

import pprint
pprint.pprint(all_scores)