from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "AAPL_LSTM_Final_compatible.keras"
)

SCALER_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "AAPL_MinMaxScaler.pkl"
)

# ------------------------------------------------------------
# LOAD MODEL AND SCALER
# ------------------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = ["Open", "High", "Low", "Close", "Volume"]
SEQUENCE_LENGTH = 60


# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No CSV file uploaded."

    file = request.files["file"]

    if file.filename == "":
        return "Please select a CSV file."

    if not file.filename.lower().endswith(".csv"):
        return "Please upload a CSV file."

    try:

        # Read CSV
        df = pd.read_csv(file)

        # Required columns
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
            if column not in df.columns
        ]

        if missing_columns:
            return f"Missing columns: {missing_columns}"

        # Date processing
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.dropna(subset=["Date"])

        df = df.sort_values("Date").reset_index(drop=True)

        # Need at least 60 trading days
        if len(df) < SEQUENCE_LENGTH:
            return "CSV must contain at least 60 trading days."

        # Take latest 60 days
        df = df.tail(SEQUENCE_LENGTH).copy()

        # Convert numerical columns
        for column in FEATURES:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        if df[FEATURES].isnull().any().any():
            return "CSV contains invalid or missing numerical values."

        # Prepare input
        input_data = df[FEATURES].values

        # Apply the same scaler used during training
        input_scaled = scaler.transform(input_data)

        # LSTM input shape: (1, 60, 5)
        X_new = input_scaled.reshape(
            1,
            SEQUENCE_LENGTH,
            len(FEATURES)
        )

        # Prediction
        prediction_scaled = model.predict(
            X_new,
            verbose=0
        )

        # Convert predicted Close back to original scale
        dummy_prediction = np.zeros(
            (1, len(FEATURES))
        )

        dummy_prediction[0, 3] = prediction_scaled[0, 0]

        predicted_price = scaler.inverse_transform(
            dummy_prediction
        )[0, 3]

        # Last actual closing price
        last_close = float(
            df["Close"].iloc[-1]
        )

        # Change
        price_change = predicted_price - last_close

        percentage_change = (
            price_change / last_close
        ) * 100

        direction = (
            "UP"
            if price_change > 0
            else "DOWN"
        )

        # Return result to HTML
        return render_template(
            "index.html",
            input_start_date=df["Date"].iloc[0].strftime("%Y-%m-%d"),
            input_end_date=df["Date"].iloc[-1].strftime("%Y-%m-%d"),
            trading_days=len(df),
            last_close=f"${last_close:.2f}",
            predicted_close=f"${predicted_price:.2f}",
            price_change=f"${price_change:.2f}",
            percentage_change=f"{percentage_change:.2f}%",
            direction=direction
        )

    except Exception as e:
        return f"Prediction error: {str(e)}"


# ------------------------------------------------------------
# RUN APPLICATION
# ------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)