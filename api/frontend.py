from fastapi import FastAPI
from nicegui import app, ui
from nicegui.events import KeyEventArguments

def init(fastapi_app: FastAPI) -> None:

    ticker_list = set()
    all_tickers = gt.get_tickers(NYSE=True, NASDAQ=True, AMEX=True)

    @ui.page('/')
    def show():
        ticker = ui.input(label="Enter a stock ticker:",
                          placeholder="e.g., AAPL",
                          validation={"Input expects a ticker symbol": lambda value: len(value) <= 4},
                          autocomplete=all_tickers
                    )

        def handle_key(e: KeyEventArguments):
            if e.key == "Enter" and e.action.keydown:
                ui.notify(f"{e.key} was just pressed!")
                ticker_list.add(ticker.value)
                print(ticker_list)

        keyboard = ui.keyboard(on_key=handle_key)

        ui.dark_mode().bind_value(app.storage.user, 'dark_mode')
        ui.checkbox('dark mode').bind_value(app.storage.user, 'dark_mode')

    ui.run_with(
        fastapi_app,
        storage_secret='PR1V4T3K3Y'
    )
