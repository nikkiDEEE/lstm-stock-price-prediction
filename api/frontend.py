from fastapi import FastAPI
from nicegui import app, ui

def init(fastapi_app: FastAPI) -> None:
    @ui.page('/')
    def show():
        ui.input(label="Enter a stock ticker:",
                placeholder="e.g., AAPL",
                on_change=lambda x: ticker.set_text("Searching for: " + x.value),
                validation={"Input expects a ticker symbol": lambda value: len(value) == 4}  
            )
        ticker = ui.label()

        ui.dark_mode().bind_value(app.storage.user, 'dark_mode')
        ui.checkbox('dark mode').bind_value(app.storage.user, 'dark_mode')

    ui.run_with(
        fastapi_app,
        storage_secret='PR1V4T3K3Y'
    )
