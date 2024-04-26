import pandas as pd
import numpy as np


def get_ticker_data(ticker):
    try:
        df = pd.read_csv("data/tickers.csv")
    except FileNotFoundError:
        print("Error: tickers.csv file not found.")
        return None, None, None
    
    ticker_row = df[df["ticker"] == ticker]

    if not ticker_row.empty:
        stock_name = ticker_row.iloc[0]['name']
        exchange = ticker_row.iloc[0]['exchange']
        return ticker, stock_name, exchange
    else:
        return None, None, None


def get_all_tickers():
    try:
        df = pd.read_csv("data/tickers.csv")
    except FileNotFoundError:
        print("Error: tickers.csv file not found.")
        return None
    
    return set(df["ticker"].tolist())
    
