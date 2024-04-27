from utils import model

def process_tickers(ticker_list):
    plots = []
    for ticker in ticker_list:
        model_obj = model.Model(ticker)
        test_preds_fig = model_obj.plot_test_preds()
        future_trend_fig = model_obj.plot_future_trend()
        plots.append((ticker, test_preds_fig, future_trend_fig))

    return plots
