from utils import model

def process_tickers(ticker_list):
    """
    Processes a list of tickers and generates plots for the test predictions and future trend for each ticker.
    
    Args:
        ticker_list (list): A list of tickers to process.
    
    Returns:
        list: A list of tuples, where each tuple contains the ticker and the corresponding test predictions and future trend plots.
        
    """
    plots = []
    for ticker in ticker_list:
        model_obj = model.Model(ticker)
        test_preds_fig = model_obj.plot_test_preds()
        future_trend_fig = model_obj.plot_future_trend()
        plots.append((ticker, test_preds_fig, future_trend_fig))

    return plots
