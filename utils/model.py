import pandas as pd
import numpy as np
import plotly.graph_objects as go

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append('/home/nikonubuntu/code/ds340/utils')

from preprocess_utils import train_test_split, get_df


class Model:
    def __init__(self, ticker, window_size=91, start_year=2000):
        self.ticker = str(ticker)
        self.window_size = int(window_size)
        self.start_year = int(start_year)

        self.df = get_df(self.ticker, self.start_year)
        self.scaler = MinMaxScaler()
        self.scaled_df = self.scale_df()
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = train_test_split(self.scaled_df, self.window_size)

        self.model = self.create_model()
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, validation_data=(self.X_val, self.y_val))
        self.loss = self.model.evaluate(self.X_test, self.y_test)


    def scale_df(self):
        df_copy = self.df.copy()
        df_copy.loc[:, 'Close'] = self.scaler.fit_transform(df_copy[['Close']]).flatten()

        return df_copy


    def create_model(self):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()
        
        return model
    

    def plot_test_preds(self):
        y_pred = self.model.predict(self.X_test)
        y_pred = self.scaler.inverse_transform(y_pred)
        y_test = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=self.df.index[-len(y_test):], y=y_pred.flatten(), mode='lines', name='Predicted Data (X_test)', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=self.df.index[-len(y_test):], y=y_test.flatten(), mode='lines', name='Actual Data (y_test)', line=dict(color='blue')))

        fig.update_layout(
            xaxis=dict(title='Date'),
            yaxis=dict(title='Close Price'),
            title='Actual vs Predicted Trend on Test Data',
            legend=dict(x=0, y=1),
            width=1200,
            height=600
        )

        return fig
    
    def get_future_window(self):
        last_window = np.array(self.scaled_df['Close'].iloc[-self.window_size:])
        preds = np.reshape(last_window, (1, self.window_size, 1))

        for _ in range(self.window_size):
            y_pred = self.model.predict(preds)
            preds = np.append(preds, y_pred.reshape(1, 1, 1), axis=1)
            preds = preds[:, 1:, :]

        return preds
    
    def plot_future_trend(self):
        future_date_range = pd.date_range(start=self.df.index[-1] + pd.Timedelta(days=1), periods=self.window_size)
        preds = self.get_future_window()
        future_df = pd.DataFrame(preds.squeeze(), index=future_date_range, columns=['Close'])
        future_df['Close'] = self.scaler.inverse_transform(future_df[['Close']])
        extended_df = pd.concat([self.df, future_df])

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=extended_df.index, y=extended_df['Close'], mode='lines', name='Predicted Data', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Close'], mode='lines', name='Historical Data', line=dict(color='blue')))

        fig.update_layout(
            xaxis=dict(title='Date'),
            yaxis=dict(title='Close Price'),
            title='Historical Data and Predicted Trend',
            legend=dict(x=0, y=1),
            width=1200,
            height=600
        )

        return fig
