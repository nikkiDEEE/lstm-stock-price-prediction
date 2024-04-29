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
        """
        Initializes the model with the given ticker, window size, and start year. 
        Retrieves the DataFrame for the given ticker, scales the 'Close' column, splits the data into training, validation, and test sets, creates the LSTM model, and trains it on the training and validation data.
        
        Args:
            ticker (str): The stock ticker to use for the model.
            window_size (int, optional): The size of the sliding window to use for the LSTM model. Defaults to 91.
            start_year (int, optional): The start year for the stock data. Defaults to 2000.
        
        Attributes:
            ticker (str): The stock ticker used for the model.
            window_size (int): The size of the sliding window used for the LSTM model.
            start_year (int): The start year for the stock data.
            df (pandas.DataFrame): The DataFrame containing the stock data.
            scaler (sklearn.preprocessing.MinMaxScaler): The scaler used to scale the 'Close' column.
            scaled_df (pandas.DataFrame): The scaled DataFrame.
            X_train (numpy.ndarray): The training input data.
            y_train (numpy.ndarray): The training output data.
            X_val (numpy.ndarray): The validation input data.
            y_val (numpy.ndarray): The validation output data.
            X_test (numpy.ndarray): The test input data.
            y_test (numpy.ndarray): The test output data.
            model (keras.models.Sequential): The LSTM model.
            loss (float): The loss of the model on the test data.

        """
        self.ticker = str(ticker)
        self.window_size = int(window_size)
        self.start_year = int(start_year)

        self.df = get_df(self.ticker, self.start_year)
        self.scaler = MinMaxScaler()
        self.scaled_df = self.scale_df()
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = train_test_split(self.scaled_df, self.window_size)

        self.model = self.create_model()
        self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, validation_data=(self.X_val, self.y_val))
        self.loss = self.model.evaluate(self.X_test, self.y_test)


    def scale_df(self):
        """
        Scales the 'Close' column of the DataFrame `df` using the `scaler` attribute.
        
        Returns:
            A copy of the DataFrame `df` with the 'Close' column scaled.

        """
        df_copy = self.df.copy()
        df_copy.loc[:, 'Close'] = self.scaler.fit_transform(df_copy[['Close']]).flatten()

        return df_copy


    def create_model(self):
        """
        Creates a sequential LSTM model for time series forecasting.
        
        The model consists of three LSTM layers with 50 units each, followed by a Dropout layer with a rate of 0.2 after each LSTM layer.
        The final layer is a Dense layer with a single unit, which outputs the predicted value.
        
        The model is compiled with the 'adam' optimizer and 'mean_squared_error' loss function.
        
        Returns:
            A compiled Keras Sequential model.

        """
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
        """
        Plots the actual vs predicted trend on the test data.
        
        Returns:
            go.Figure: A Plotly figure object containing the plot of the actual and predicted data on the test set.

        """
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
        """
        Generates a window of future predictions based on the last window of the input data.
        
        Returns:
            np.ndarray: A 3D numpy array containing the predicted future window. The shape of the array is (1, window_size, 1), where window_size is the size of the prediction window.

        """
        last_window = np.array(self.scaled_df['Close'].iloc[-self.window_size:])
        preds = np.reshape(last_window, (1, self.window_size, 1))

        for _ in range(self.window_size):
            y_pred = self.model.predict(preds)
            preds = np.append(preds, y_pred.reshape(1, 1, 1), axis=1)
            preds = preds[:, 1:, :]

        return preds
    
    def plot_future_trend(self):
        """
        Plots a figure that displays the historical data and the predicted future trend.
        
        Returns:
            go.Figure: A Plotly figure object containing the plot of the historical data and the predicted future trend.
            
        """
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
