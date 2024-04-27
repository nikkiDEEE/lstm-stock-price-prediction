import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit

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


def get_df(symbol, start_year=1980):
    raw_df = yf.download(symbol, interval='1d')
    df_copy = raw_df.copy()
    df_copy = df_copy[['Close']]
    df_copy = df_copy[df_copy.index.year >= start_year]

    return df_copy


def train_test_split(df, window_size):
    X, y = [], []

    for i in range(window_size, len(df)):
        feature = np.array(df['Close'].iloc[i - window_size:i])
        target = np.array(df['Close'].iloc[i])

        X.append(feature)
        y.append(target)

    X, y = np.array(X), np.array(y)

    tscv = TimeSeriesSplit()

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_val_split = int(len(train_index) * 0.8)
        X_train, X_val = X_train[:train_val_split], X_train[train_val_split:]
        y_train, y_val = y_train[:train_val_split], y_train[train_val_split:]

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test
