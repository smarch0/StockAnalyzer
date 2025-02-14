from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import pytz
import yfinance as yf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def get_est_timestamp(utc_timestamp: pd.Timestamp) -> str:
    """
    Convert UTC timestamp to EST and format it as a string.
    """
    try:
        return (
            utc_timestamp.to_pydatetime()
            .replace(tzinfo=pytz.utc)
            .astimezone(pytz.timezone("US/Eastern"))
            .strftime("%Y-%m-%d %H:%M:%S %Z")
        )
    except Exception as e:
        logger.error(f"Error converting timestamp: {e}")
        return "N/A"

def clean_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names by extracting only the first element if they are tuples.
    """
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    return data

def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).
    """
    if "Close" not in data:
        logger.error("Dataframe missing required column: Close")
        return pd.Series(dtype=float)
    
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_sma(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    """
    if "Close" not in data:
        logger.error("Dataframe missing required column: Close")
        return pd.Series(dtype=float)
    return data["Close"].rolling(window=window).mean()

def get_stock_data(ticker: str) -> dict[str, Any]:
    """
    Fetch stock data and compute technical indicators.
    """
    try:
        data = yf.download(ticker, period="2d", interval="5m", prepost=True)
        if data.empty:
            logger.warning(f"No data retrieved for {ticker}.")
            return {}

        data = clean_column_names(data)  # Normalize column names
        data.dropna(inplace=True)
        
        data["RSI"] = calculate_rsi(data)
        data["SMA10"] = calculate_sma(data, 10)
        data["SMA50"] = calculate_sma(data, 50)
        data["SMA200"] = calculate_sma(data, 200)
        data["Timestamp"] = data.index.map(get_est_timestamp)
        data.reset_index(drop=True, inplace=True)

        save_to_csv(ticker, data.iloc[-1].to_dict())

        return data.iloc[-1].to_dict()
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return {}

def save_to_csv(ticker: str, stock_data: dict[str, Any]) -> None:
    """
    Save stock data to CSV.
    """
    if not stock_data:
        logger.warning(f"No data to save for {ticker}.")
        return
    
    filename = f"{ticker}_stock_data.csv"
    headers = [str(key) for key in stock_data.keys()]
    
    try:
        file_exists = os.path.exists(filename)
        with open(filename, "a") as f:
            if not file_exists:
                f.write(",".join(headers) + "\n")
            f.write(",".join(map(str, stock_data.values())) + "\n")
        logger.info(f"Data saved: {stock_data}")
    except Exception as e:
        logger.error(f"Error saving data for {ticker}: {e}")

def create_candlestick_chart(data: pd.DataFrame, ticker: str) -> None:
    """
    Create a candlestick chart.
    """
    if data.empty:
        logger.warning(f"No data available for candlestick chart: {ticker}")
        return
    
    try:
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"]
            )
        ])

        fig.update_layout(
            title=f"{ticker} Candlestick Chart",
            xaxis_title="Time",
            yaxis_title="Price",
        )
        fig.show()
    except ImportError:
        logger.error("Plotly not installed.")
    except Exception as e:
        logger.error(f"Error creating chart for {ticker}: {e}")

def main() -> None:
    """
    Main execution function.
    """
    tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
    # for _ in range(3):  # while loop if you want to run a few times
    for ticker in tech_stocks:
        logger.info(f"Fetching stock data for {ticker}...")
        stock_data = get_stock_data(ticker)
        save_to_csv(ticker, stock_data)

if __name__ == "__main__":
    main()
