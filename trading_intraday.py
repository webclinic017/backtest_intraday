from datetime import datetime
from threading import Timer
from pytz import timezone
import pandas as pd
import logging
from joblib import load

from data_fetcher import get_alpaca_stocks_and_save, get_alpaca_account_data, get_market_clock, subscribe_to_stream, \
    get_existing_position_in_ticker, get_existing_positions, submit_limit_order
from indicators import normalize_columns_with_predefined_scaler, columns_to_normalize
from stock_utils import get_only_trading_hours_from_df_dict, apply_features_for_stocks, get_indicators_for_df, \
    get_signals_for_df, update_n_minute_bar, get_last_row_action_from_stock, get_live_positions_value, \
    get_live_positions_ticker_names, close_position, get_stock_quantity_to_trade, get_leveraged_etf_price
from strategies import calculate_returns_for_df_based_on_signals_alone
from utils import get_next_period_minute_window_date

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


# TODO: what I should do for a complete software:
'''
1. Every week on sunday, the backtest will run.
What will be saved: 
    a. the model. override last for now. V
    b. the raw + engineered csvs of all stocks. override last for now. V
    c. adjustment ratio per stock. override last for now. FOR NOW, I Don't care about adjusted data since I only use ETFs and not actual stocks. V

2. Every Monday-Friday at 9:25 EST, start running the trading script.
    a. get today's market hours
        i) if there are no market hours, finish the script with log - "US market not open today" V
        iia) if there are market hours, set up 2 timeouts
            - one for timeToOpen - trigger the script in the callback V
            - one for timeToClose - trigger close all streams, save whats needed and delete variables that could explode over time. add a log "Done trading for today". Could be nice if there was a log to telegram
        iib) since there is no good way to implement timeouts in python, pull every 5 minutes, check the times. X
    b. every minute or connect to a stream, and pull the latest 5-minute candle for each stock
        i) for each new candle in stock, connect it to latest stock's raw df and indicators df.
        ii) save the new raw stock.
        iii) add indicators for the last line of the stock.
        iv) save the indicators df.
        v) check if we are currently in a trade for this stock.
'''

all_stocks_dict_with_features = {}
model = None
last_train_scaler = None


def check_exit_position(position_data):
    return None


def trade_row(row_data, ticker):
    return None


def handle_last_row_action(action, price, ticker, current_positions, account_data, prediction):
    live_positions_value = get_live_positions_value(current_positions)
    live_positions_ticker_names = get_live_positions_ticker_names(current_positions)
    current_cash = account_data.cash
    if 'Exit' in action:
        if ticker not in live_positions_ticker_names:
            print(f'Warning: Found exit alert but not a live position in {ticker}')
            return None
        else:
            return close_position(ticker, price)
    elif 'Bullish' in action or 'Bearish' in action:
        if ticker in live_positions_ticker_names:
            print(f'Warning: Found trade alert while I already have a live position in {ticker}, action: {action}, positions: {current_positions}')
            return None
        else:
            # TODO: for now, predict last row and pront prediction - but later, we should use the prediction to determine if we should enter trade or not.
            # TODO: for now, lets try to maintain a 20% equity per position
            current_leveraged_etf_name, current_leveraged_etf_price = get_leveraged_etf_price(ticker)
            print(f'Trade alert, ticker: {ticker}, action: {action}, prediction: {prediction}')
            stock_quantity = get_stock_quantity_to_trade(live_positions_value, current_leveraged_etf_price, current_cash, 0.2)
            return submit_limit_order(current_leveraged_etf_name, current_leveraged_etf_price, action, stock_quantity)
    return None


def create_mock_bar():
    return {
        't': '2022-05-28T04:18:29.933217006-04:00',
        'o': 140,
        'h': 150,
        'l': 138,
        'c': 143,
        'v': 10000,
        'S': 'symbol'
    }


async def on_new_stock_data(bar):
    current_account_data = get_alpaca_account_data()
    # Not sure if bar is a dict or a dataframe or a series.
    print(f'Received new stock data {bar["S"]}')
    # TODO: 1. save as last row on all_stocks_dict_with_features with new_column "Real_Date" and save 5 minute after last row on "Date"
    #  2. add indicators, actions and all to the last row on all_stocks_dict_with_features
    #  3. check for new trade for stock
    #  4. on new bar data, check if last row on all_stocks_dict_with_features has a "Real_Date" that is equal to its "Date", if so repeat 1-3,
    #  5. else, "merge" the last row with the current data to create a new candle and repeat 2 & 3.
    ticker_name = bar["S"]
    bar_date = datetime.fromtimestamp(bar['t'].seconds, tz=timezone('US/Eastern'))
    bar = pd.DataFrame(bar, index=[0])
    bar = bar.reset_index().rename(
        columns={'t': 'Real_Date', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close',
                 'v': 'Volume', 'S': 'symbol'})
    bar = bar[['Real_Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']]
    bar['Real_Date'] = bar_date
    bar['Date'] = get_next_period_minute_window_date(5, bar_date)

    global all_stocks_dict_with_features
    global last_train_scaler
    global model

    bar = update_n_minute_bar(all_stocks_dict_with_features[ticker_name].iloc[-1], bar, 5)

    if bar.at[0, 'Real_Date'] != bar.at[0, 'Date'] and 'Real_Date' in all_stocks_dict_with_features[ticker_name].columns:
        all_stocks_dict_with_features[ticker_name] = all_stocks_dict_with_features[ticker_name].iloc[:-1]
    all_stocks_dict_with_features[ticker_name] = pd.concat([all_stocks_dict_with_features[ticker_name], bar])
    all_stocks_dict_with_features[ticker_name] = all_stocks_dict_with_features[ticker_name].reset_index(drop=True)
    all_stocks_dict_with_features[ticker_name] = get_indicators_for_df(all_stocks_dict_with_features[ticker_name], ticker_name)
    all_stocks_dict_with_features[ticker_name] = get_signals_for_df(all_stocks_dict_with_features[ticker_name], ticker_name)
    all_stocks_dict_with_features[ticker_name] = calculate_returns_for_df_based_on_signals_alone(
            all_stocks_dict_with_features[ticker_name], ticker_name)
    all_stocks_dict_with_features[ticker_name] = normalize_columns_with_predefined_scaler(
        all_stocks_dict_with_features[ticker_name], columns_to_normalize, last_train_scaler)
    # TODO: 1. check for exit flag in stock
    #  2. if exit flag, check if we are currently in a position for this stock (or leveraged stock).
    #  3. if we are in a position, exit the position.
    #  4. if we are not in a position, log warning and do nothing.
    # TODO: 1. check for new trade flag in stock.
    #  2. if new trade flag, check if we are currently in a position for this stock (or leveraged stock).
    #  3. if we are in a position, log warning and do nothing.
    #  4. if we are not in a position, check if I have 20% of my total assets (cash+equity) as available cash.
    #  5. if I have enough cash, enter a new position for this stock.
    #  6. if I do not have enough cash, check if this position is better than other positions I currently have (don't develop yet - this is a complex feature that takes in consideration scoring (which I dont have), am I profitable in the position im currently checking?, am i about to be profitable in the position im currently checking?, etc).
    #  7. if the position is not better than other positions I currently have, log info and do nothing.
    #  8. if the position is better than other positions I currently have, exit the worst position (don't develop yet) and enter to this position.
    #  9. missing a case of buy that turns into sell and vice versa. Don't think I have implemented such a case in the first place.
    last_row_action, last_row_price, last_row_prediction = get_last_row_action_from_stock(all_stocks_dict_with_features[ticker_name].iloc[-1], ticker_name, model)
    all_positions = get_existing_positions()
    result = handle_last_row_action(last_row_action, last_row_price, ticker_name, all_positions, current_account_data, last_row_prediction)
    return result


# async def subscribe_wrapper(stocks_dict, scaler, callback):
#     callback(stocks_dict, scaler)


def prepare_initial_trading_day_data(tickers, last_train_scaler):
    global all_stocks_dict_with_features
    stocks_dict = get_alpaca_stocks_and_save(tickers)
    stocks_dict = get_only_trading_hours_from_df_dict(stocks_dict, tickers)
    all_stocks_dict_with_features = apply_features_for_stocks(stocks_dict, tickers)
    for ticker in tickers:
        all_stocks_dict_with_features[ticker] = calculate_returns_for_df_based_on_signals_alone(
            all_stocks_dict_with_features[ticker], ticker)
        all_stocks_dict_with_features[ticker] = normalize_columns_with_predefined_scaler(
            all_stocks_dict_with_features[ticker], columns_to_normalize, last_train_scaler)
    return all_stocks_dict_with_features


def on_market_open(tickers):
    print(f'Start trading for today {datetime}')
    # TODO: trigger trading script
    global all_stocks_dict_with_features
    global last_train_scaler
    global model
    last_train_scaler = load('last_train_scaler.gz')
    model = load('intraday_model.joblib')
    all_stocks_dict_with_features = prepare_initial_trading_day_data(tickers, last_train_scaler)
    subscribe_to_stream(tickers, on_new_stock_data)
    # TODO: create a function that aggregates one minute data from stream and construct a 5 minute data candle from it per stock.
    return None


def on_market_close():
    print(f'Done trading for today {datetime}')
    # TODO: trigger save all thats needed, close all streams, delete all global vars, print account data, print todays p&l
    return None


def init_trading(tickers):
    account = get_alpaca_account_data()
    print(account)
    market_times = get_market_clock()
    print(f'today\'s market times: {market_times}')

    if not market_times.is_open:
        if market_times.next_open.date() != datetime.today().date():
            print('markets not open today')
            return
        eastern_tz = timezone('US/Eastern')
        now_eastern = pd.to_datetime(datetime.now(eastern_tz))
        next_open_eastern = market_times.next_open.astimezone(eastern_tz)
        next_close_eastern = market_times.next_close.astimezone(eastern_tz)
        timeout_to_open = Timer((next_open_eastern - now_eastern).total_seconds(), on_market_open, ([tickers]))
        timeout_to_close = Timer((next_close_eastern - now_eastern).total_seconds(), on_market_close, ([]))
        timeout_to_open.start()
        timeout_to_close.start()
    # TODO: just for now testing. delete the following line call to on_market_open()!
    on_market_open(tickers)


'''
I have a few ways to tackle this:
1. Stream:
    a. timeout to open
    b. get all initial data from the past few days
    c. subscribe to stream of each stock
    d. on new stock data, concat it to previous data
    d. process trading on modified data.
    e. finish work on timeout to close (don't forget to close streams)
2. Interval:
    a. scheduler start interval (1 minute) on market open
    b. on each interval: get all data from the past few days
    c. check for new data per stock (compare with previous run)
    d. process trading per new data per stock
    e. finish work on timeout to close (don't forget to cancel scheduled job)
'''