# AAPL Stock Price Prediction Using LSTM

## Project Overview

This project uses a Long Short-Term Memory (LSTM) neural network to predict the **next trading day's closing price of Apple Inc. (AAPL)** based on historical stock market data.

The model learns patterns from the previous **60 trading days** using Open, High, Low, Close, and Volume values.

The project includes:

- LSTM model training
- Time-series preprocessing
- Model evaluation
- Saved trained model and scaler
- CSV-based prediction system
- Next trading-day closing price prediction

---

## Objective

To develop a deep learning model that can analyze historical AAPL stock price data and predict the closing price for the next trading day.

### Input

The system requires the latest **60 trading days** containing:

- Date
- Open
- High
- Low
- Close
- Volume

### Output

The LSTM model predicts:

**Next trading day's closing price**

---

## Dataset

The historical AAPL stock dataset was sourced from Yahoo Finance through Kaggle.

**Kaggle Dataset:**  
https://www.kaggle.com/datasets/farhanali097/apple-aapl-stock-data-1980-to-december-2024

### Original Dataset

- Period: 1980–2024
- Records: 11,084

### Data Used for Training

The project uses data from:

**2015-01-01 to 2024-12-31**

Actual available data:

- Start: 2015-01-02
- End: 2024-11-29
- Records: 2,495

### Features

| Feature | Description |
|---|---|
| Open | Opening stock price |
| High | Highest price during the trading day |
| Low | Lowest price during the trading day |
| Close | Closing stock price |
| Volume | Number of shares traded |

---

## Methodology

The project follows the following pipeline:

```text
Historical AAPL Dataset
        ↓
Select 2015–2024 Data
        ↓
Chronological Train/Validation/Test Split
        ↓
Feature Selection
        ↓
MinMaxScaler Normalization
        ↓
Create 60-Day Sequences
        ↓
LSTM Model Training
        ↓
Model Evaluation
        ↓
Save Model + Scaler
        ↓
Upload New 60-Day CSV
        ↓
Predict Next Trading Day Closing Price
