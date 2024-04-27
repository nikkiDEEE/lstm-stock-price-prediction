from fastapi import FastAPI
from nicegui import app, ui

from backend import process_tickers
from utils.preprocess_utils import get_ticker_data, get_all_tickers

all_tickers = get_all_tickers()

def handle_add_click(ticker, ticker_list, ticker_display):
    ticker = (ticker.value).upper()
    _, stock_name, exchange = get_ticker_data(ticker)
    if stock_name == None or exchange == None:
        ui.notify(f"{ticker} is not a valid ticker symbol!")
    else:
        ui.notify(f"{ticker} - {stock_name} - {exchange} added!")
        ticker_list.add(ticker)
        ticker_display.set_text("\n".join(ticker_list))


def init(fastapi_app: FastAPI) -> None:

    ticker_list = set()

    @ui.page('/')
    def show():
        ticker = ui.input(label="Enter a ticker symbol:",
                          placeholder="e.g., AAPL",
                          validation={"Input expects a ticker symbol": lambda value: value.upper() in all_tickers},
                    )

        ticker_display = ui.label("[ Tickers added appear here ]")

        ui.button("Add ticker", on_click=lambda: handle_add_click(ticker, ticker_list, ticker_display))

        def handle_submit_click(ticker_list):
            plots = process_tickers(list(ticker_list))
            for ticker, test_preds_fig, future_trend_fig in plots:
                with ui.row():
                    ui.label(f"{ticker} - Test Predictions").classes('text-lg font-bold')
                    ui.plotly(test_preds_fig)
                    ui.plotly(future_trend_fig)

        ui.button("Submit", on_click=handle_submit_click(ticker_list))

        ui.dark_mode().auto()

    ui.run_with(
        fastapi_app,
        storage_secret='PR1V4T3K3Y'
    )
