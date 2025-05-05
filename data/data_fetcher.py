# data/data_fetcher.py

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

def fetch_data(ticker_list, start_date, end_date):
    return YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=ticker_list
    ).fetch_data()
