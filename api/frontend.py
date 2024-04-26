from fastapi import FastAPI
from nicegui import app, ui
from backend import process_tickers

def handle_add_click(ticker, ticker_list, ticker_display):
    ticker_value = (ticker.value).upper()
    ui.notify(f"{ticker_value} added!")
    ticker_list.add(ticker_value)
    ticker_display.set_text("\n".join(sorted(ticker_list)))

def handle_submit_click(ticker_list):
    ui.notify("Submitting tickers...")
    process_tickers(ticker_list)

def init(fastapi_app: FastAPI) -> None:

    ticker_list = set()

    @ui.page('/')
    def show():
        ticker = ui.input(label="Enter a ticker symbol:",
                          placeholder="e.g., AAPL",
                          validation={"Input expects a ticker symbol": lambda value: len(value) <= 4},
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
