import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, RMSprop, SGD, Adamax
from statsmodels.tsa.arima.model import ARIMA
from joblib import dump, load
from datetime import timedelta

# Define model names and create directories
MODEL_NAMES = ['RandomForest', 'XGBoost', 'LinearRegression', 'ARIMA', 'LSTM']
for model in MODEL_NAMES:
    os.makedirs(f"trained_models/{model}", exist_ok=True)
    os.makedirs(f"charts/{model}", exist_ok=True)
os.makedirs("charts/comparison", exist_ok=True)
os.makedirs("charts/LSTM/optimizers", exist_ok=True)
os.makedirs("charts/forecast", exist_ok=True)

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

def plot_predictions(df, actual, train_pred, test_pred, title, output_path, seq_len=50):
    plt.figure(figsize=(15, 8))
    plt.plot(df.index[seq_len:], actual[seq_len:], label="Actual Prices", color='green')
    plt.plot(df.index[seq_len:seq_len+len(train_pred)], train_pred, label="Train Predictions", color='orange')
    plt.plot(df.index[seq_len+len(train_pred):seq_len+len(train_pred)+len(test_pred)], test_pred, label="Test Predictions", color='teal')
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_training_history(history, model_name, stock_name):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{stock_name} - {model_name} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f"charts/{model_name}/{stock_name}_training_history.png")
    plt.close()

def plot_optimizer_comparison(optimizer_scores, stock_name):
    plt.figure(figsize=(12, 6))
    metrics = ['MAE', 'MSE', 'RMSE', 'R2']
    for metric in metrics:
        values = [scores[metric] for opt, scores in optimizer_scores.items()]
        plt.plot(values, label=metric)
    plt.xticks(range(len(optimizer_scores)), optimizer_scores.keys())
    plt.title(f'{stock_name} - LSTM Optimizer Comparison')
    plt.xlabel('Optimizer')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"charts/LSTM/optimizers/{stock_name}_optimizer_comparison.png")
    plt.close()

def plot_model_comparison(all_scores, metric, output_path, stock_name="HPG"):
    plt.figure(figsize=(8, 6))
    stock_names = list(all_scores.keys())
    
    if not stock_names:
        print(f"No data to plot for {metric}")
        plt.close()
        return

    scores = all_scores[stock_names[0]]
    model_names = []
    values = []

    for model_name in MODEL_NAMES:
        score = scores.get(model_name)
        if isinstance(score, dict) and metric in score:
            model_names.append(model_name)
            values.append(score[metric])

    if not values:
        print(f"No valid data to plot for {metric}")
        plt.close()
        return

    plt.bar(model_names, values, color=['blue', 'orange', 'red', 'green', 'purple'])
    plt.title(f'Model Comparison - {metric} ({stock_name})')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_forecast_1_day(df, actual, forecast_1, model_name, stock_name, last_date):
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, actual, label="Historical Prices", color='green')
    future_dates_1 = [last_date + timedelta(days=1)]
    plt.plot(future_dates_1, [forecast_1], 'ro', label="1-Day Forecast")
    plt.legend()
    plt.title(f"{stock_name} - {model_name} Price Forecast (1 Day)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"charts/forecast/{stock_name}_{model_name}_forecast_1_day.png")
    plt.close()

def plot_forecast_15_days(df, actual, forecast_15, model_name, stock_name, last_date):
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, actual, label="Historical Prices", color='green')
    future_dates_15 = [last_date + timedelta(days=i+1) for i in range(15)]
    plt.plot(future_dates_15, forecast_15, 'b-', label="15-Day Forecast")
    plt.legend()
    plt.title(f"{stock_name} - {model_name} Price Forecast (15 Days)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"charts/forecast/{stock_name}_{model_name}_forecast_15_days.png")
    plt.close()

def plot_forecast_30_days(df, actual, forecast_30, model_name, stock_name, last_date):
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, actual, label="Historical Prices", color='green')
    future_dates_30 = [last_date + timedelta(days=i+1) for i in range(30)]
    plt.plot(future_dates_30, forecast_30, 'purple', label="30-Day Forecast")
    plt.legend()
    plt.title(f"{stock_name} - {model_name} Price Forecast (30 Days)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"charts/forecast/{stock_name}_{model_name}_forecast_30_days.png")
    plt.close()

def create_sequences(data, seq_len, forecast_steps=1, target_col_idx=3):  # Default target is 'Close'
    X, y = [], []
    for i in range(len(data) - seq_len - forecast_steps + 1):
        X.append(data[i:i+seq_len])  # All features for seq_len time steps
        y.append(data[i+seq_len:i+seq_len+forecast_steps, target_col_idx])  # Target is 'Close'
    return np.array(X), np.array(y)

def calculate_trend_slope(close_prices, window=30):
    if len(close_prices) < window:
        return 0
    recent_prices = close_prices[-window:]
    x = np.arange(len(recent_prices))
    slope, _ = np.polyfit(x, recent_prices, 1)
    slope = np.clip(slope, -0.005, 0.005)
    return slope

def estimate_arima_order(data, max_p=5, max_d=2, max_q=5):
    d = 0
    temp_data = data.copy()
    while d <= max_d:
        if d > 0:
            temp_data = np.diff(temp_data)
        if len(temp_data) > 1:
            variance = np.var(temp_data)
            if variance < 0.01 or d == max_d:
                break
        d += 1
    p, q = 1, 1  # Simplified; consider using auto_arima for better order selection
    return p, d, q

def train_and_forecast(filepath, seq_len=50, forecast_horizons=[1, 15, 30]):
    # Load and preprocess data
    df = load_json_file(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.dropna()

    if df.shape[0] < seq_len + max(forecast_horizons):
        print(f"Skipping {filepath}: Insufficient data (<{seq_len + max(forecast_horizons)} rows).")
        return None

    # Define feature columns
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'SMA_5', 'SMA_20', 'EMA_5', 'EMA_20', 
                       'RSI_14', 'MACD', 'Signal_Line', 
                       'Upper_Band', 'Lower_Band', 'ATR_14', 'OBV']
    
    # Apply log transformation to price-related features
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                  'SMA_5', 'SMA_20', 'EMA_5', 'EMA_20', 
                  'Upper_Band', 'Lower_Band']
    for col in price_cols:
        df[col] = np.log1p(df[col].clip(lower=0))  # Ensure non-negative values

    # Scale all features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])
    dump(scaler, f"trained_models/scaler_{os.path.basename(filepath).replace('.json', '')}.joblib")

    # Create sequences
    X_scaled, y_scaled = create_sequences(scaled_data, seq_len, forecast_steps=1, target_col_idx=3)  # Close is at index 3
    print("X shape:",X_scaled.shape)
    print("Y shape", y_scaled.shape)
    # Split into train and test sets
    train_size = int(len(X_scaled) * 0.8)
    X_train_scaled, X_test_scaled = X_scaled[:train_size], X_scaled[train_size:]
    y_train_scaled, y_test_scaled = y_scaled[:train_size], y_scaled[train_size:]

    model_scores = {}
    stock_name = os.path.basename(filepath).replace('.json', '')
    last_date = df.index[-1]
    last_sequence_scaled = scaled_data[-seq_len:]  # All features for the last sequence

    # Calculate trend slope and volatility based on Close prices
    close_prices = np.log1p(df['Close'].values)
    trend_slope = calculate_trend_slope(close_prices, window=30)
    volatility = np.std(close_prices[-30:]) / np.mean(close_prices[-30:]) if len(close_prices) >= 30 else 0.01

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=4, random_state=42)
    rf.fit(X_train_scaled.reshape(X_train_scaled.shape[0], -1), y_train_scaled.flatten())
    dump(rf, f"trained_models/RandomForest/{stock_name}.joblib")
    y_rf_train_scaled = rf.predict(X_train_scaled.reshape(X_train_scaled.shape[0], -1))
    y_rf_test_scaled = rf.predict(X_test_scaled.reshape(X_test_scaled.shape[0], -1))
    
    # Inverse transform predictions
    y_rf_train = scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_rf_train_scaled), len(feature_columns)-1)), 
                        y_rf_train_scaled.reshape(-1, 1)], axis=1))[:, 3]
    y_rf_test = scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_rf_test_scaled), len(feature_columns)-1)), 
                        y_rf_test_scaled.reshape(-1, 1)], axis=1))[:, 3]
    y_rf_train = np.expm1(y_rf_train)
    y_rf_test = np.expm1(y_rf_test)
    y_test_original = np.expm1(scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_test_scaled), len(feature_columns)-1)), 
                        y_test_scaled.reshape(-1, 1)], axis=1))[:, 3])
    
    model_scores["RandomForest"] = evaluate_model(y_test_original, y_rf_test, "Random Forest Test")
    plot_predictions(df, df['Close'], y_rf_train, y_rf_test, 
                     f"{stock_name} - Random Forest", f"charts/RandomForest/{stock_name}.png", seq_len)

    # Random Forest Forecast
    forecasts_rf = {}
    current_sequence = last_sequence_scaled.copy()
    for horizon in forecast_horizons:
        predictions = []
        temp_sequence = current_sequence.copy()
        for i in range(horizon):
            pred_scaled = rf.predict(temp_sequence.reshape(1, -1))
            pred = scaler.inverse_transform(
                np.concatenate([np.zeros((1, len(feature_columns)-1)), 
                                pred_scaled.reshape(-1, 1)], axis=1))[:, 3]
            pred = np.expm1(pred)[0]
            pred = max(0, pred + (1 - np.exp(-0.1 * i)) * trend_slope + np.random.normal(0, volatility * pred))
            predictions.append(pred)
            # Update sequence with predicted Close price (simplified)
            scaled_pred = scaler.transform(
                np.concatenate([temp_sequence[-1, :-1].reshape(1, -1), 
                                np.log1p([[pred]]).reshape(1, -1)], axis=1))
            temp_sequence = np.roll(temp_sequence, -1, axis=0)
            temp_sequence[-1] = scaled_pred
        forecasts_rf[f"{horizon}_days"] = predictions
    plot_forecast_1_day(df, df['Close'], forecasts_rf["1_days"][0], "RandomForest", stock_name, last_date)
    plot_forecast_15_days(df, df['Close'], forecasts_rf["15_days"], "RandomForest", stock_name, last_date)
    plot_forecast_30_days(df, df['Close'], forecasts_rf["30_days"], "RandomForest", stock_name, last_date)

    # XGBoost
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, min_child_weight=3, reg_lambda=1.5, random_state=42)
    xgb.fit(X_train_scaled.reshape(X_train_scaled.shape[0], -1), y_train_scaled.flatten())
    dump(xgb, f"trained_models/XGBoost/{stock_name}.joblib")
    y_xgb_train_scaled = xgb.predict(X_train_scaled.reshape(X_train_scaled.shape[0], -1))
    y_xgb_test_scaled = xgb.predict(X_test_scaled.reshape(X_test_scaled.shape[0], -1))
    
    y_xgb_train = scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_xgb_train_scaled), len(feature_columns)-1)), 
                        y_xgb_train_scaled.reshape(-1, 1)], axis=1))[:, 3]
    y_xgb_test = scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_xgb_test_scaled), len(feature_columns)-1)), 
                        y_xgb_test_scaled.reshape(-1, 1)], axis=1))[:, 3]
    y_xgb_train = np.expm1(y_xgb_train)
    y_xgb_test = np.expm1(y_xgb_test)
    
    model_scores["XGBoost"] = evaluate_model(y_test_original, y_xgb_test, "XGBoost Test")
    plot_predictions(df, df['Close'], y_xgb_train, y_xgb_test, 
                     f"{stock_name} - XGBoost", f"charts/XGBoost/{stock_name}.png", seq_len)

    # XGBoost Forecast
    forecasts_xgb = {}
    current_sequence = last_sequence_scaled.copy()
    for horizon in forecast_horizons:
        predictions = []
        temp_sequence = current_sequence.copy()
        for i in range(horizon):
            pred_scaled = xgb.predict(temp_sequence.reshape(1, -1))
            pred = scaler.inverse_transform(
                np.concatenate([np.zeros((1, len(feature_columns)-1)), 
                                pred_scaled.reshape(-1, 1)], axis=1))[:, 3]
            pred = np.expm1(pred)[0]
            pred = max(0, pred + (1 - np.exp(-0.1 * i)) * trend_slope + np.random.normal(0, volatility * pred))
            predictions.append(pred)
            scaled_pred = scaler.transform(
                np.concatenate([temp_sequence[-1, :-1].reshape(1, -1), 
                                np.log1p([[pred]]).reshape(1, -1)], axis=1))
            temp_sequence = np.roll(temp_sequence, -1, axis=0)
            temp_sequence[-1] = scaled_pred
        forecasts_xgb[f"{horizon}_days"] = predictions
    plot_forecast_1_day(df, df['Close'], forecasts_xgb["1_days"][0], "XGBoost", stock_name, last_date)
    plot_forecast_15_days(df, df['Close'], forecasts_xgb["15_days"], "XGBoost", stock_name, last_date)
    plot_forecast_30_days(df, df['Close'], forecasts_xgb["30_days"], "XGBoost", stock_name, last_date)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled.reshape(X_train_scaled.shape[0], -1), y_train_scaled.flatten())
    dump(lr, f"trained_models/LinearRegression/{stock_name}.joblib")
    y_lr_train_scaled = lr.predict(X_train_scaled.reshape(X_train_scaled.shape[0], -1))
    y_lr_test_scaled = lr.predict(X_test_scaled.reshape(X_test_scaled.shape[0], -1))
    
    y_lr_train = scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_lr_train_scaled), len(feature_columns)-1)), 
                        y_lr_train_scaled.reshape(-1, 1)], axis=1))[:, 3]
    y_lr_test = scaler.inverse_transform(
        np.concatenate([np.zeros((len(y_lr_test_scaled), len(feature_columns)-1)), 
                        y_lr_test_scaled.reshape(-1, 1)], axis=1))[:, 3]
    y_lr_train = np.expm1(y_lr_train)
    y_lr_test = np.expm1(y_lr_test)
    
    model_scores["LinearRegression"] = evaluate_model(y_test_original, y_lr_test, "Linear Regression Test")
    plot_predictions(df, df['Close'], y_lr_train, y_lr_test, 
                     f"{stock_name} - Linear Regression", f"charts/LinearRegression/{stock_name}.png", seq_len)

    # Linear Regression Forecast
    forecasts_lr = {}
    current_sequence = last_sequence_scaled.copy()
    for horizon in forecast_horizons:
        predictions = []
        temp_sequence = current_sequence.copy()
        for i in range(horizon):
            pred_scaled = lr.predict(temp_sequence.reshape(1, -1))
            pred = scaler.inverse_transform(
                np.concatenate([np.zeros((1, len(feature_columns)-1)), 
                                pred_scaled.reshape(-1, 1)], axis=1))[:, 3]
            pred = np.expm1(pred)[0]
            pred = max(0, pred + (1 - np.exp(-0.1 * i)) * trend_slope + np.random.normal(0, volatility * pred))
            predictions.append(pred)
            scaled_pred = scaler.transform(
                np.concatenate([temp_sequence[-1, :-1].reshape(1, -1), 
                                np.log1p([[pred]]).reshape(1, -1)], axis=1))
            temp_sequence = np.roll(temp_sequence, -1, axis=0)
            temp_sequence[-1] = scaled_pred
        forecasts_lr[f"{horizon}_days"] = predictions
    plot_forecast_1_day(df, df['Close'], forecasts_lr["1_days"][0], "LinearRegression", stock_name, last_date)
    plot_forecast_15_days(df, df['Close'], forecasts_lr["15_days"], "LinearRegression", stock_name, last_date)
    plot_forecast_30_days(df, df['Close'], forecasts_lr["30_days"], "LinearRegression", stock_name, last_date)

    # ARIMA
    try:
        p, d, q = estimate_arima_order(close_prices)
        arima = ARIMA(close_prices, order=(p, d, q))
        arima_model = arima.fit()
        y_arima_pred = arima_model.predict(start=train_size+seq_len, end=len(close_prices)-1, type='levels')  # Updated 'typ' to 'type'
        y_arima_pred = np.expm1(y_arima_pred)
        model_scores["ARIMA"] = evaluate_model(y_test_original, y_arima_pred, "ARIMA Test")
        plot_predictions(df, df['Close'], df['Close'].values[seq_len:seq_len+train_size], y_arima_pred, 
                         f"{stock_name} - ARIMA", f"charts/ARIMA/{stock_name}.png", seq_len)

        forecasts_arima = {}
        for horizon in forecast_horizons:
            forecast = arima_model.forecast(steps=horizon)
            forecast = np.expm1(forecast)
            forecast = [max(0, forecast[i] + (1 - np.exp(-0.1 * i)) * trend_slope + np.random.normal(0, volatility * forecast[i])) for i in range(len(forecast))]
            forecasts_arima[f"{horizon}_days"] = forecast
        plot_forecast_1_day(df, df['Close'], forecasts_arima["1_days"][0], "ARIMA", stock_name, last_date)
        plot_forecast_15_days(df, df['Close'], forecasts_arima["15_days"], "ARIMA", stock_name, last_date)
        plot_forecast_30_days(df, df['Close'], forecasts_arima["30_days"], "ARIMA", stock_name, last_date)
    except Exception as e:
        print(f"ARIMA failed for {stock_name}: {str(e)}")
        model_scores["ARIMA"] = f"ARIMA Failed: {str(e)}"
        forecasts_arima = {f"{horizon}_days": [] for horizon in forecast_horizons}

    # LSTM
    try:
        X_train_lstm = X_train_scaled  # Shape: (samples, seq_len, n_features)
        X_test_lstm = X_test_scaled

        optimizers = {
            'adam': Adam(learning_rate=0.001),
            'rmsprop': RMSprop(learning_rate=0.001),
            'sgd': SGD(learning_rate=0.001, momentum=0.9),
            'adamax': Adamax(learning_rate=0.001)
        }
        
        best_lstm_score = float('inf')
        best_optimizer = None
        best_model = None
        best_history = None
        optimizer_scores = {}
        
        for opt_name, optimizer in optimizers.items():
            model_lstm = Sequential([
                LSTM(128, return_sequences=True, input_shape=(seq_len, len(feature_columns))),
                Dropout(0.3),
                LSTM(64, return_sequences=True),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.3),
                Dense(16),
                Dense(1)
            ])
            model_lstm.compile(optimizer=optimizer, loss='mse')

            es = EarlyStopping(patience=10, restore_best_weights=True)
            history = model_lstm.fit(
                X_train_lstm, 
                y_train_scaled, 
                epochs=100, 
                batch_size=32, 
                validation_split=0.2, 
                callbacks=[es], 
                verbose=1
            )
            
            y_pred_lstm_scaled = model_lstm.predict(X_test_lstm)
            y_pred_lstm = scaler.inverse_transform(
                np.concatenate([np.zeros((len(y_pred_lstm_scaled), len(feature_columns)-1)), 
                                y_pred_lstm_scaled], axis=1))[:, 3]
            y_pred_lstm = np.expm1(y_pred_lstm)
            y_test_lstm = scaler.inverse_transform(
                np.concatenate([np.zeros((len(y_test_scaled), len(feature_columns)-1)), 
                                y_test_scaled], axis=1))[:, 3]
            y_test_lstm = np.expm1(y_test_lstm)
            
            scores = evaluate_model(y_test_lstm, y_pred_lstm, f"LSTM ({opt_name}) Test")
            optimizer_scores[opt_name] = scores
            
            mse = mean_squared_error(y_test_lstm, y_pred_lstm)
            if mse < best_lstm_score:
                best_lstm_score = mse
                best_optimizer = opt_name
                best_model = model_lstm
                best_history = history

        plot_optimizer_comparison(optimizer_scores, stock_name)
        model_scores["LSTM"] = optimizer_scores[best_optimizer]
        best_model.save(f"trained_models/LSTM/{stock_name}.keras")
        plot_training_history(best_history, "LSTM", stock_name)

        y_train_lstm_scaled = best_model.predict(X_train_lstm)
        y_train_lstm = scaler.inverse_transform(
            np.concatenate([np.zeros((len(y_train_lstm_scaled), len(feature_columns)-1)), 
                            y_train_lstm_scaled], axis=1))[:, 3]
        y_train_lstm = np.expm1(y_train_lstm)
        plot_predictions(df, df['Close'], y_train_lstm, y_pred_lstm, 
                         f"{stock_name} - LSTM ({best_optimizer})", f"charts/LSTM/{stock_name}.png", seq_len)

        # LSTM Forecast
        forecasts_lstm = {}
        current_sequence = last_sequence_scaled.reshape((1, seq_len, len(feature_columns)))
        for horizon in forecast_horizons:
            predictions = []
            temp_sequence = current_sequence.copy()
            for i in range(horizon):
                pred_scaled = best_model.predict(temp_sequence, verbose=0)
                pred = scaler.inverse_transform(
                    np.concatenate([np.zeros((1, len(feature_columns)-1)), 
                                    pred_scaled], axis=1))[:, 3]
                pred = np.expm1(pred)[0]
                pred = max(0, pred + (1 - np.exp(-0.1 * i)) * trend_slope + np.random.normal(0, volatility * pred))
                predictions.append(pred)
                scaled_pred = scaler.transform(
                    np.concatenate([temp_sequence[0, -1, :-1].reshape(1, -1), 
                                    np.log1p([[pred]]).reshape(1, -1)], axis=1))
                temp_sequence = np.roll(temp_sequence, -1, axis=1)
                temp_sequence[0, -1, :] = scaled_pred[0]
            forecasts_lstm[f"{horizon}_days"] = predictions
        plot_forecast_1_day(df, df['Close'], forecasts_lstm["1_days"][0], "LSTM", stock_name, last_date)
        plot_forecast_15_days(df, df['Close'], forecasts_lstm["15_days"], "LSTM", stock_name, last_date)
        plot_forecast_30_days(df, df['Close'], forecasts_lstm["30_days"], "LSTM", stock_name, last_date)
    except Exception as e:
        print(f"LSTM failed for {stock_name}: {str(e)}")
        model_scores["LSTM"] = f"LSTM Failed: {str(e)}"
        forecasts_lstm = {f"{horizon}_days": [] for horizon in forecast_horizons}

    return model_scores, {"RandomForest": forecasts_rf, "XGBoost": forecasts_xgb, 
                         "LinearRegression": forecasts_lr, "ARIMA": forecasts_arima, "LSTM": forecasts_lstm}

# Process a single file
file_path = "/kaggle/input/stock-1142025/processed_stock_data_scaled/HPG.json"
all_scores = {}
all_forecasts = {}

if os.path.exists(file_path):
    print(f"Processing {file_path}...")
    df = load_json_file(file_path)
    print(f"Loaded data with shape: {df.shape}")
    if df.shape[0] < 51:
        print(f"Skipping {file_path}: Insufficient data (<51 rows).")
    else:
        result, forecasts = train_and_forecast(file_path, seq_len=50)
        if result:
            all_scores[os.path.basename(file_path)] = result
            all_forecasts[os.path.basename(file_path)] = forecasts
        else:
            print(f"No results returned for {file_path}.")
else:
    print(f"File {file_path} does not exist.")

# Generate comparison plots
for metric in ['MAE', 'MSE', 'RMSE', 'R2']:
    plot_model_comparison(all_scores, metric, f"charts/comparison/{metric}_comparison.png", stock_name="HPG")

import pprint
pprint.pprint(all_scores)
pprint.pprint(all_forecasts)
