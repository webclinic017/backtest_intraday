#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import asyncio
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from alpaca_trade_api.stream import Stream
from alpha_vantage.async_support.timeseries import TimeSeries
import os
import numpy as np
import pandas as pd
import requests
import time
import io

from utils import save_create_csv
from utils import get_todays_start_of_trade_str, read_stock_from_file, save_create_csv, convert_action_to_api_action

key_path = '/Users/yochainusan/Desktop/backtest_intraday/config/alpha_vantage/key.txt' # TODO: delete once I implemented dotenv
telegram_key_path = '/Users/yochainusan/Desktop/backtest_intraday/config/telegram/key.txt' # TODO: delete once I implemented dotenv

api = REST(api_version='v2')
stream = Stream(data_feed='sip', raw_data=True)
# stream = Stream(data_feed='sip', raw_data=True)


def get_existing_position_in_ticker(ticker):
    return api.get_position(ticker)


def get_existing_positions():
    # returns an empty array [] if no positions exist
    return api.list_positions()


def submit_limit_order(ticker, price, action, stock_quantity):
    api_action = convert_action_to_api_action(action)
    return api.submit_order(
        symbol=ticker,
        side=api_action,
        type='limit',
        qty=stock_quantity,
        time_in_force='day',
        order_class='simple',
        limit_price=price,
        # TODO: order_class='bracket' is used for setting up stop loss &/or take profit orders
        # order_class='bracket',
        # take_profit=dict(
        #     limit_price='305.0',
        # ),
        # stop_loss=dict(
        #     stop_price='295.5',
        #     limit_price='295.5',
        # )
    )



def subscribe_to_stream(tickers, on_new_data_callback):
    print('--- subscribing to stream ---')
    for ticker in tickers:
        stream.subscribe_bars(on_new_data_callback, ticker)
        # stream.subscribe_updated_bars(on_new_data_callback, ticker)
        # stream.subscribe_news(on_new_data_callback, ticker)
    stream.run()


def retry_get_request(url):
    retries = 1
    success = False
    while not success and retries <= 10:
        try:
            response = requests.get(url)
            success = True
            return response
        except Exception as e:
            wait = retries * 10
            print(f'Error Get Requesting {url}: {e}')
            print('Error! Waiting %s secs and re-trying...' % wait)
            time.sleep(wait)
            retries += 1
    return None


def get_sp500_list():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    df.to_csv('S&P500-Info.csv')
    df.to_csv("S&P500-Symbols.csv", columns=['Symbol'])
    return df['Symbol']


def get_alpaca_account_data():
    print('--- getting alpaca account data ---')
    account = api.get_account()
    return account


def get_market_clock():
    print('--- getting market clock ---')
    market_times = api.get_clock()
    return market_times


def get_alpaca_stock_latest_bar(ticker):
    frame = TimeFrame(1, TimeFrameUnit.Minute)
    bars = api.get_bars(ticker, frame)
    return bars[-1]


def get_alpaca_stocks_and_save(tickers):
    print('--- getting alpaca stocks ---')
    frame = TimeFrame(5, TimeFrameUnit.Minute)
    today_trade_start = get_todays_start_of_trade_str()
    data_rows_num = 78 * 2 * len(tickers) # 2 days of data per ticker
    all_tickers_df = api.get_bars(tickers, frame, start=today_trade_start, limit=data_rows_num).df
    all_tickers_df = all_tickers_df.tz_convert('US/Eastern')
    all_tickers_df = all_tickers_df.reset_index().rename(columns={'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    recent_stocks_dict = {}
    for ticker_name, stock_df in all_tickers_df.groupby('symbol'):
        print(stock_df)
        recent_stocks_dict[ticker_name] = stock_df
        save_create_csv('recent_stocks_raw_data', ticker_name, stock_df)
    return recent_stocks_dict


def get_stock_earnings_data(ticker, start_time, time):
    print('--- getting earnings data ---')
    print(ticker)
    api_key = open(key_path,'r').read()
    r = requests.get(f'https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={api_key}')
    json_result = r.json()
    time.sleep(0.4 - ((time.time() - start_time) % 0.4))
    return json_result


def convert_columns_to_adjusted(stock_df):
    df = stock_df.copy()
    adjustment_ratio = df['Adjusted_Close'] / df['Close']
    df['Open'] = (adjustment_ratio * df['Open']).round(2)
    df['High'] = (adjustment_ratio * df['High']).round(2)
    df['Low'] = (adjustment_ratio * df['Low']).round(2)
    df['Close'] = (df['Adjusted_Close']).round(2)
    df = df.drop(['Adjusted_Close'], axis=1)
    return df


def get_stock_data_trade_daily_alpha_vantage_csv(ticker):
    print(ticker)
    api_key = open(key_path, 'r').read()
    data = retry_get_request(f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={api_key}&outputsize=full&datatype=csv').content
    if data is None:
        return None
    data = pd.read_csv(io.StringIO(data.decode('utf-8')))
    if 'close' not in data:
        return None
    data = data[['timestamp', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']]
    data.columns = ["Date", "Open","High","Low","Close","Adjusted_Close","Volume"]
    data = data.iloc[::-1].reset_index(drop=True)
    data = data.iloc[-(252*8):].reset_index(drop=True) # TODO: 4 years of data
    df = convert_columns_to_adjusted(data)
    # time.sleep(0.4 - ((time.time() - start_time) % 0.4))
    return df


async def get_stock_data_trade_daily_alpha_vantage(ticker):
    print(ticker)
    api_key = open(key_path, 'r').read()

    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = await ts.get_daily_adjusted(ticker, outputsize='full')
    await ts.close()

    if data is None:
        return None
    if '4. close' not in data:
        return None
    data['timestamp'] = data.index
    data = data[['timestamp', '1. open', '2. high', '3. low', '4. close', '5. adjusted close', '6. volume']]
    data.columns = ["Date", "Open","High","Low","Close","Adjusted_Close","Volume"]
    data = data.iloc[::-1].reset_index(drop=True)
    data = data.iloc[-(252*4):].reset_index(drop=True) # TODO: 4 years of data [[[[ should be -252*4 ]]]]
    df = convert_columns_to_adjusted(data)
    return ticker, df


def retry_read_csv_from_remote(num_retries, url):
    retries = 1
    success = False
    while not success and retries <= num_retries:
        try:
            data = retry_get_request(url).content
            data = pd.read_csv(io.StringIO(data.decode('utf-8')))
            success = True
            return data
        except Exception as e:
            wait = retries * 3
            print(f'Error Get Requesting And Reading {url}: {e}')
            print('Error! Waiting %s secs and re-trying...' % wait)
            time.sleep(wait)
            retries += 1
    return None


def get_stock_data_intraday_alpha_vantage(ticker):
    print(ticker)
    api_key = open(key_path, 'r').read()

    aggregated_df = pd.DataFrame()
    for i in range(1, 25):
        print(f'month {i}')

        data = retry_read_csv_from_remote(10, f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={ticker}&apikey={api_key}&interval=5min&slice=year{1 if i <= 12 else 2}month{i if i <= 12 else i - 12}&outputsize=full')
        if data is None:
            return None
        # data = data.set_index('time')
        # data.index = pd.to_datetime(data.index)
        # data = data.between_time('9:30', '16:00')
        # data = data.reset_index()
        aggregated_df = aggregated_df.append(data)
    if aggregated_df is None:
        return None
    if 'close' not in aggregated_df:
        return None
    aggregated_df = aggregated_df[['time', 'open', 'high', 'low', 'close', 'volume']]
    aggregated_df.columns = ["Date", "Open","High","Low","Close","Volume"]
    dataframe = aggregated_df.iloc[::-1].reset_index(drop=True)
    return dataframe


def get_data_for_stock(ticker, interval, start_time, time_module):
    # switch call between daily adjusted, TODO: intraday_extended! and weekly adjusted
    if interval == 'D':
        stock_data = get_stock_data_trade_daily_alpha_vantage(ticker)
        if stock_data is None:
            return None
        return stock_data


def add_earnings_dates_to_stock(stock_df, earnings_json):
    df = stock_df.copy()
    df['is_earning_days'] = None
    for quarterly_report in earnings_json['quarterlyEarnings']:
        report_date_string = quarterly_report['reportedDate']
        df['is_earning_days'][df['Date'] == report_date_string] = True
    return df['is_earning_days']


def get_data_dict_for_multiple_stocks(tickers, time_module):
    ohlc_intraday = {} # dictionary with ohlc value for each stock
    # api_usage_limit_per_minute = 150
    api_usage_limit_per_minute = 1

    for ticker in tickers:
        # TODO: toggle comment between these 2 lines for daily/intraday data
        ohlc_intraday[ticker] = get_stock_data_intraday_alpha_vantage(ticker)
        # ohlc_intraday[ticker] = get_stock_data_trade_daily_alpha_vantage_csv(ticker)
        save_create_csv('stocks_csvs_raw', ticker, ohlc_intraday[ticker])
    return ohlc_intraday


def get_data_dict_for_all_stocks_in_directory(directory_str):
    directory = os.fsencode(directory_str)
    ohlc_intraday = {}
    tickers = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and filename[0].isupper():
            ticker = filename.split('.csv')[0].split('_')[0]
            print(f'pulling ticker csv {ticker}')
            stock_df = pd.read_csv(directory_str + '/' + filename)
            ohlc_intraday[ticker] = stock_df[['Date','Open','High','Low','Close','Volume']]
            tickers.append(ticker)
    return ohlc_intraday, tickers


def get_dfs_for_all_csvs_in_directory(directory_str):
    directory = os.fsencode(directory_str)
    df_list = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            df = pd.read_csv(directory_str + '/' + filename)
            df_list.append(df)
    return df_list
