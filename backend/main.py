from fastapi import FastAPI

app = FastAPI(
    title="AAPL Stock Prediction API",
    description="FastAPI backend for AAPL LSTM stock price prediction",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "AAPL LSTM Stock Prediction API is running"
    }
