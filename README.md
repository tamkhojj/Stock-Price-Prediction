# 📈 Stock Price Prediction on VNIndex (2017–2025)

Welcome to our stock price prediction project, where we apply cutting-edge machine learning and deep learning techniques to forecast stock prices of **374 companies listed on the VNIndex**, covering a comprehensive period from **2017 to 2025**.

This project aims to build accurate and interpretable models for time series prediction, catering to both financial analysis and academic research.

---

## 🚀 Overview

In the dynamic and often unpredictable world of the stock market, having robust forecasting models can offer a major edge. This project explores different predictive approaches to estimate **future closing prices** of stocks listed on the Vietnamese stock market (VNIndex).

Each model is trained and evaluated individually per stock using a unified and preprocessed dataset, ensuring fair comparison and scalability.

---

## ✨ Features

- 📅 **Historical price data** from 2017 to 2025 for 374 stocks.
- 📊 **Technical indicators** included: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV.
- ⚙️ **Five powerful prediction models** implemented and benchmarked.
- 📉 **Visualization** of real vs predicted prices for each stock and model.
- 📦 Modular structure for scalability and further research.
- 📁 Trained models and evaluation scores are saved automatically.

---

## 🧠 Algorithms Used

The project leverages a combination of **traditional statistical methods**, **machine learning models**, and **deep learning architectures**:

| Model              | Type            | Description |
|-------------------|------------------|-------------|
| **ARIMA**         | Statistical      | Autoregressive Integrated Moving Average, effective for stationary time series. |
| **Random Forest** | Machine Learning | Ensemble of decision trees for robust, non-linear regression. |
| **XGBoost**       | Machine Learning | Gradient boosting framework known for high accuracy and efficiency. |
| **Linear Regression** | Machine Learning | Simple yet effective baseline for trend modeling. |
| **LSTM**          | Deep Learning    | Recurrent neural network ideal for time series with long-term dependencies. |

---

## 📚 Dataset

- **Source**: Custom-collected and preprocessed dataset from VNIndex.
- **Format**: Each stock is stored in a separate `.json` file.
- **Fields per record**:
  - `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
  - Technical indicators: `SMA`, `EMA`, `RSI`, `MACD`, `Signal_Line`, `Upper_Band`, `Lower_Band`, `ATR_14`, `OBV`
- **Total Stocks**: 374  
- **Time Range**: 2017 to 2025

---
