from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import io


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AAPL Stock Prediction API",
    description="LSTM-based next trading day stock price prediction API",
    version="1.0.0"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "AAPL_LSTM_Final.keras"
)

SCALER_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "AAPL_MinMaxScaler.pkl"
)


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

SEQUENCE_LENGTH = 60


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "AAPL LSTM Stock Prediction API is running"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # --------------------------------------------------------
    # Check file type
    # --------------------------------------------------------

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:

        # ----------------------------------------------------
        # Read uploaded CSV
        # ----------------------------------------------------

        contents = await file.read()

        input_df = pd.read_csv(
            io.BytesIO(contents)
        )

        # ----------------------------------------------------
        # Validate required columns
        # ----------------------------------------------------

        required_columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in input_df.columns
        ]

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing_columns}"
            )

        # ----------------------------------------------------
        # Convert Date
        # ----------------------------------------------------

        input_df["Date"] = pd.to_datetime(
            input_df["Date"],
            errors="coerce"
        )

        input_df = input_df.dropna(
            subset=["Date"]
        )

        # ----------------------------------------------------
        # Sort chronologically
        # ----------------------------------------------------

        input_df = input_df.sort_values(
            "Date"
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # Check number of records
        # ----------------------------------------------------

        if len(input_df) < SEQUENCE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain at least 60 trading days."
            )

        # ----------------------------------------------------
        # Use latest 60 trading days
        # ----------------------------------------------------

        input_df = input_df.tail(
            SEQUENCE_LENGTH
        ).copy()

        # ----------------------------------------------------
        # Check numeric values
        # ----------------------------------------------------

        for column in FEATURES:
            input_df[column] = pd.to_numeric(
                input_df[column],
                errors="coerce"
            )

        if input_df[FEATURES].isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail="CSV contains invalid or missing numeric values."
            )

        # ----------------------------------------------------
        # Prepare input data
        # ----------------------------------------------------

        input_data = input_df[FEATURES].values

        # ----------------------------------------------------
        # Apply saved scaler
        # ----------------------------------------------------

        input_scaled = scaler.transform(
            input_data
        )

        # ----------------------------------------------------
        # Reshape for LSTM
        # ----------------------------------------------------

        X_new = input_scaled.reshape(
            1,
            SEQUENCE_LENGTH,
            len(FEATURES)
        )

        # ----------------------------------------------------
        # Predict next closing price
        # ----------------------------------------------------

        prediction_scaled = model.predict(
            X_new,
            verbose=0
        )

        # ----------------------------------------------------
        # Inverse transform prediction
        # ----------------------------------------------------

        dummy_prediction = np.zeros(
            (1, len(FEATURES))
        )

        dummy_prediction[0, 3] = (
            prediction_scaled[0, 0]
        )

        predicted_price = scaler.inverse_transform(
            dummy_prediction
        )[0, 3]

        # ----------------------------------------------------
        # Calculate price change
        # ----------------------------------------------------

        last_close = float(
            input_df["Close"].iloc[-1]
        )

        price_change = (
            predicted_price - last_close
        )

        percentage_change = (
            price_change / last_close
        ) * 100

        direction = (
            "UP"
            if price_change > 0
            else "DOWN"
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {
            "input_start_date": input_df["Date"].iloc[0].strftime("%Y-%m-%d"),
            "input_end_date": input_df["Date"].iloc[-1].strftime("%Y-%m-%d"),
            "trading_days": len(input_df),
            "last_close": round(last_close, 2),
            "predicted_close": round(float(predicted_price), 2),
            "price_change": round(float(price_change), 2),
            "percentage_change": round(float(percentage_change), 2),
            "direction": direction
        }


# ============================================================
# END
# ============================================================
