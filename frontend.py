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

def handle_submit_click(ticker_list):
    ui.notify("Submitting tickers...")
    process_tickers(ticker_list)
    # ticker_values = get_ticker_values(ticker_list)  # Get values for tickers
    # results_page.set_values(ticker_values)  # Pass values to results page
    # app.router.navigate(results_page)  # Navigate to results page

def init(fastapi_app: FastAPI) -> None:

    ticker_list = set()

    @ui.page('/')
    def show():
        ticker = ui.input(label="Enter a ticker symbol:",
                          placeholder="e.g., AAPL",
                          validation={"Input expects a ticker symbol": lambda value: value.upper() in all_tickers},
                          autocomplete=list(all_tickers)
                    )

        ticker_display = ui.label("[ Tickers added appear here ]")

        ui.button("Add ticker", on_click=lambda: handle_add_click(ticker, ticker_list, ticker_display))
        ui.button("Submit", on_click=lambda: handle_submit_click(ticker_list))

        ui.dark_mode().bind_value(app.storage.user, 'dark_mode')
        ui.checkbox('dark mode').bind_value(app.storage.user, 'dark_mode')

    ui.run_with(
        fastapi_app,
        storage_secret='PR1V4T3K3Y'
    )
