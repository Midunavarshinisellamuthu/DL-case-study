# AAPL Stock Price Prediction Using LSTM

## Project Overview

This project uses a Long Short-Term Memory (LSTM) neural network
to predict the next trading day's closing price of Apple Inc. (AAPL).

## Dataset

Historical AAPL stock data sourced from Yahoo Finance through Kaggle.
Link: https://www.kaggle.com/datasets/farhanali097/apple-aapl-stock-data-1980-to-december-2024

- Original period: 1980–2024
- Data used: 2015–2024
- Features:
  - Open
  - High
  - Low
  - Close
  - Volume

## Methodology

1. Load historical AAPL stock data
2. Select data from 2015–2024
3. Perform chronological train/validation/test split
4. Normalize features using MinMaxScaler
5. Create 60-day sliding-window sequences
6. Train an LSTM neural network
7. Evaluate using MAE, RMSE and MAPE
8. Predict the next trading day's closing price

## Model Architecture

Input
↓
LSTM (50 units)
↓
Dropout (0.2)
↓
Dense (1)
↓
Predicted Closing Price

## Results

| Metric | Result |
|---|---:|
| MAE | 4.5535 |
| RMSE | 5.6357 |
| MAPE | 2.1512% |

## Next-Day Prediction

Last available date: 2024-11-29

Last actual closing price: $237.33

Predicted next closing price: $229.17

## Technologies

- Python
- Google Colab
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
