import datetime

import numpy as np
import pandas as pd

from data_fetcher import submit_limit_order, get_alpaca_stock_latest_bar
from indicators import get_ma_column_for_stock, simple_slope, get_breakout_column_for_stock, \
    get_touch_and_return_above_column_for_stock, get_macd_columns_for_stock, get_ATR_column_for_stock, \
    get_distance_between_columns_for_stock, get_adx_column_for_stock, rsi, stochastic, get_volatility_from_atr
from signals import crossing_mas
from utils import save_create_csv, get_leveraged_etfs, get_feature_col_names


def get_only_trading_hours_from_df_dict(dfs_dict, tickers):
    for ticker in tickers:
        start = datetime.time(9, 35)
        end = datetime.time(16, 5)
        dfs_dict[ticker]['Date'] = pd.to_datetime(dfs_dict[ticker]['Date'])
        dfs_dict[ticker] = dfs_dict[ticker][dfs_dict[ticker]['Date'].dt.time.between(start, end)]
        dfs_dict[ticker] = dfs_dict[ticker].reset_index()
    return dfs_dict


def get_live_positions_value(positions):
    positions_value = 0
    for position in positions:
        positions_value += abs(float(position.market_value))
    return positions_value


def get_stock_quantity_to_trade(live_positions_value, price, current_cash, pct_of_total_equity):
    total_equity = current_cash + live_positions_value
    ideal_cash_to_trade = total_equity * pct_of_total_equity
    actual_cash_to_trade = min(ideal_cash_to_trade, current_cash)
    if actual_cash_to_trade < ideal_cash_to_trade:
        print("Actual cash to trade is less than ideal cash to trade. Ideal cash to trade: {}, actual cash to trade: {}".format(ideal_cash_to_trade, actual_cash_to_trade))
    quantity_to_trade = int(actual_cash_to_trade / price)
    if quantity_to_trade < 1:
        print("Quantity to trade is less than 1")
    return quantity_to_trade


def close_position(position_data, limit_price):
    if position_data.side == 'long':
        # sell the stock
        return submit_limit_order(position_data.symbol, 'Exit Buy', limit_price, position_data.qty)
    elif position_data.side == 'short':
        # buy the stock
        return submit_limit_order(position_data.symbol, 'Exit Sell', limit_price, position_data.qty)


def get_live_positions_ticker_names(positions):
    ticker_names = []
    leveraged_etfs = get_leveraged_etfs()
    for position in positions:
        leveraged_etfs_dict = next(item for item in leveraged_etfs if item["1x"] == position.symbol or item["2x"] == position.symbol or item["3x"] == position.symbol)
        ticker_names += leveraged_etfs_dict.values()
    return ticker_names


def convert_action_to_api_action(order):
    if order == 'Bullish' or order == 'Exit Sell':
        return 'buy'
    elif order == 'Bearish' or order == 'Exit Buy':
        return 'sell'


def update_n_minute_bar(stock_df_last_row, current_bar, minute_period):
    if 'Real_Date' not in stock_df_last_row or stock_df_last_row['Real_Date'].minute % minute_period == 0:
        return current_bar
    current_bar.at[0, 'Open'] = stock_df_last_row['Open']
    current_bar.at[0, 'High'] = max(stock_df_last_row['High'], current_bar.at[0, 'High'])
    current_bar.at[0, 'Low'] = min(stock_df_last_row['Low'], current_bar.at[0, 'Low'])
    current_bar.at[0, 'Volume'] = stock_df_last_row['Volume'] + current_bar.at[0, 'Volume']
    return current_bar


def get_indicators_for_df(df, ticker):
    print(f'getting indicators for {ticker}')
    df['5_ma'] = get_ma_column_for_stock(df, 'Close', 5)
    df['8_ma'] = get_ma_column_for_stock(df, 'Close', 8)
    df['13_ma'] = get_ma_column_for_stock(df, 'Close', 13)
    df['5_ma_slope'] = simple_slope(df, '5_ma', 3)
    df['8_ma_slope'] = simple_slope(df, '8_ma', 3)
    df['13_ma_slope'] = simple_slope(df, '13_ma', 3)
    df['5_ma_volume'] = get_ma_column_for_stock(df, 'Volume', 5)
    df['8_ma_volume'] = get_ma_column_for_stock(df, 'Volume', 8)
    df['13_ma_volume'] = get_ma_column_for_stock(df, 'Volume', 13)
    df['5_ma_volume_break'] = get_breakout_column_for_stock(df, 'Volume', '5_ma_volume', '5_ma_volume_break')
    df['8_ma_volume_break'] = get_breakout_column_for_stock(df, 'Volume', '8_ma_volume', '8_ma_volume_break')
    df['13_ma_volume_break'] = get_breakout_column_for_stock(df, 'Volume', '13_ma_volume', '13_ma_volume_break')
    df['5_ma_touch'] = get_touch_and_return_above_column_for_stock(df, 'Close', '5_ma', '5_ma_touch', 4)
    df['8_ma_touch'] = get_touch_and_return_above_column_for_stock(df, 'Close', '8_ma', '8_ma_touch', 4)
    df['13_ma_touch'] = get_touch_and_return_above_column_for_stock(df, 'Close', '13_ma', '13_ma_touch', 4)
    # df['10_beta_SPY'] = get_beta_column(df, stocks_dict['SPY'], 10) # too long to process
    # df['50_beta_SPY'] = get_beta_column(df, stocks_dict['SPY'], 50) # too long to process
    # df['10_beta_QQQ'] = get_beta_column(df, stocks_dict['QQQ'], 10)
    # df['50_beta_QQQ'] = get_beta_column(df, stocks_dict['QQQ'], 50)
    # df['10_beta_IWM'] = get_beta_column(df, stocks_dict['IWM'], 10)
    # df['50_beta_IWM'] = get_beta_column(df, stocks_dict['IWM'], 50)
    df['median'] = (df['High'] + df['Low']) / 2
    df['ma_med_5'] = get_ma_column_for_stock(df, 'median', 5)
    df['ma_med_34'] = get_ma_column_for_stock(df, 'median', 34)
    df['awesome_osc'] = df['ma_med_5'] - df['ma_med_34']
    df['median_ratio'] = df['median'] / df['Close']
    df['ma_med_5_ratio'] = df['ma_med_5'] / df['Close']
    df['ma_med_34_ratio'] = df['ma_med_34'] / df['Close']
    df['macd'], df['macd_signal'] = get_macd_columns_for_stock(df, 12, 26, 9)
    df['atr'] = get_ATR_column_for_stock(df, 14)
    df['distance_from_5_ma'] = get_distance_between_columns_for_stock(df, 'Close', '5_ma')
    df['adx'], df['+di'], df['-di'] = get_adx_column_for_stock(df, 14)
    df['adx_ma_med_5_rat'] = df['adx'] * df['ma_med_5_ratio']
    df['rsi'] = rsi(df, 14)  # changed from 14
    df['stochastic_k'], df['stochastic_d'] = stochastic(df, 14, 3)
    df['atr_volatility'], df['atr_volatility_ma'] = get_volatility_from_atr(df, 14)
    return df


def get_signals_for_df(df, ticker):
    print(f'getting signals for {ticker}')
    df['signal_type'] = None
    df['signal_direction'] = None
    # signal_type and signal_direction columns are the columns that determine the actual orders!
    # TODO: depending on the type of signal I want, toggle comments
    # df = awesome_oscilator(df, 'signal_direction', 'signal_type')
    df = crossing_mas(df, 'signal_direction', 'signal_type')
    return df


def apply_features_for_stocks(all_stocks_dict, tickers):
    for ticker in tickers:
        print(f'applying features for full stock {ticker}')
        all_stocks_dict[ticker] = get_indicators_for_df(all_stocks_dict[ticker], ticker)
        all_stocks_dict[ticker] = get_signals_for_df(all_stocks_dict[ticker], ticker)
        save_create_csv('full_stocks_csvs_with_features', f'{ticker}',  all_stocks_dict[ticker])
    return all_stocks_dict


def get_leveraged_etf_price(ticker):
    leveraged_etfs = get_leveraged_etfs()
    corresponding_leveraged_etf = None
    for etf in leveraged_etfs:
        if etf['1x'] == ticker:
            corresponding_leveraged_etf = etf.values()[-1]
            break
    corresponding_leveraged_etf_price = get_alpaca_stock_latest_bar(corresponding_leveraged_etf).c
    return corresponding_leveraged_etf, corresponding_leveraged_etf_price


def get_last_row_action_from_stock(last_row, ticker, model):
    print(f'getting last row action for {ticker}')

    if last_row['signal'] == 'Bullish' or last_row['signal'] == 'Bearish':
        feature_col_names = get_feature_col_names()
        last_row_as_array = last_row[feature_col_names].values
        prediction = model.predict(last_row_as_array)[0]
        return last_row['signal'], last_row['entry_price'], prediction
    if last_row['exits'] == 'Exit Buy' or last_row['exits'] == 'Exit Sell':
        return last_row['exits'], last_row['Close'], 0
    return None, None, 0
