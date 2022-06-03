import math
import os
from datetime import datetime, timedelta
import pandas as pd
import pandas_market_calendars as mcal

def save_create_csv(dir_name, file_name, df):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    fullname = f'{os.path.join(dir_name, file_name)}.csv'
    df.to_csv(fullname, index=False)


def is_business_day(date):
    return bool(len(pd.bdate_range(start=date, end=date, tz='US/Eastern')))


def is_early_close_business_day(date):
    nyse = mcal.get_calendar('NYSE')
    market_hours_df = nyse.schedule(start_date=date, end_date=date)
    if market_hours_df['market_close'][0].hour < 20: # in Grinich time (+4 from NY time)
        return True
    return False


def get_todays_start_of_trade_str():
    # 2022-03-25T09:30:00Z
    # hour_str = 'T9:30:00Z' # for some reason the input time is in Grinich timezone. need to add 4 hours to this hour for now
    # TODO: because of Krakow - removing one hour: 13:30 => 12:30 - this has to be dynamic by timezone!!!
    hour_str = 'T13:30:00Z'
    today_date_str = datetime.today().strftime('%Y-%m-%d')
    # TODO: because of Saturday - subtracting one day. has to be dynamic
    # today_date_str = (datetime.today() - timedelta(1)).strftime('%Y-%m-%d')
    return today_date_str + hour_str


def get_next_period_minute_window_date(minute_period, date):
    if date.minute % minute_period == 0:
        return date
    else:
        return date + pd.Timedelta(minutes=minute_period - date.minute % minute_period)


def get_leveraged_etfs():
    # TODO: I should manage a better list than that. each dict should hold a "underlying_etf", "leveraged_etfs" (should research shortable etfs), "inverse_etfs" (should buy them in case of bearish flag)
    # TODO: I should add DOW, also to the backtest
    leveraged_tickers = [{
        '1x': 'SPY',
        'leveraged': 'SPXL',
        'inverse_leveraged': 'SPXU'
    }, {
        '1x': 'QQQ',
        'leveraged': 'TQQQ',
        'inverse_leveraged': 'SQQQ'
    }, {
        '1x': 'IWM',
        'leveraged': 'URTY',
        'inverse_leveraged': 'SRTY'
    }, {
        '1x': 'XLF',
        'leveraged': 'FAS',
        'inverse_leveraged': 'FAZ'
    }, {
        '1x': 'XLE',
        'leveraged': 'OILU',
        'inverse_leveraged': 'ERY'
    }, {
        '1x': 'XLU',
        'leveraged': 'UTSL',
        'inverse_leveraged': 'SDP'
    }, {
        '1x': 'XLV',
        'leveraged': 'CURE',
        'inverse_leveraged': 'RXD'
    }, {
        '1x': 'XLI',
        'leveraged': 'DUSL',
        'inverse_leveraged': 'SIJ'
    }, {
        '1x': 'XLP',
        'leveraged': 'XLP',
        'inverse_leveraged': 'SZK'
    }]

    # leveraged_tickers = [{
    #     '1x': 'SPY',
    #     '2x': 'SSO',
    #     '3x': 'SPXL',
    #     '-3x': 'SPXU',
    # }, {
    #     '1x': 'QQQ',
    #     '2x': 'QLD',
    #     '3x': 'TQQQ',
    #     '-3x': 'SQQQ',
    # }, {
    #     '1x': 'IWM',
    #     '2x': 'UWM',
    #     '3x': 'URTY',
    #     '-3x': 'SRTY',
    # }, {
    #     '1x': 'XLF',
    #     '3x': 'FAS',
    #     '-3x': 'FAZ',
    # }, {
    #     '1x': 'XLE',
    #     '2x': 'ERX',
    #     '3x': 'OILU',
    #     '-2x': 'ERY',
    # }, {
    #     '1x': 'XLU',
    #     '3x': 'UTSL',
    #     '-2x': 'SDP',
    # }, {
    #     '1x': 'XLV',
    #     '3x': 'CURE',
    #     '-2x': 'RXD',
    # }, {
    #     '1x': 'XLI',
    #     '3x': 'DUSL',
    #     '-2x': 'SIJ',
    # }, {
    #     '1x': 'XLP',
    #     '-2x': 'SZK'
    # }]

    return leveraged_tickers


def transformation_sin_cos(column):
  max_value = column.max()
  sin_values = [math.sin((2 * math.pi * x) / max_value) for x in list(column)]
  cos_values = [math.cos((2 * math.pi * x) / max_value) for x in list(column)]
  return sin_values, cos_values


def convert_action_to_api_action(order):
    if order == 'Bullish' or order == 'Exit Sell':
        return 'buy'
    elif order == 'Bearish' or order == 'Exit Buy':
        return 'sell'


def get_feature_col_names():
    return ['Volume_norm', '13_ma_norm', '13_ma_slope_norm', '13_ma_volume_norm', 'median_ratio_norm',
                                'ma_med_34_ratio_norm', 'awesome_osc_norm', 'macd_norm', 'macd_signal_norm',
                                'distance_from_5_ma_norm', 'adx_norm', '+di_norm', '-di_norm', 'rsi_norm',
                                'stochastic_d_norm', 'atr_volatility_ma_norm', 'binary_signal', 'binary_5_ma_vol_break',
                                'binary_5_ma_touch', 'day_of_week_sin', 'day_of_week_cos', 'ticker_sin', 'ticker_cos', 'time_of_day']


def read_stock_from_file(ticker_name, dir_name):
    directory = os.fsencode(dir_name)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and filename == ticker_name + ".csv":
            df = pd.read_csv(dir_name + '/' + filename)
            return df
    return None