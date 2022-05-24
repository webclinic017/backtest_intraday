import os
from datetime import datetime
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
    hour_str = 'T12:30:00Z'
    today_date_str = datetime.today().strftime('%Y-%m-%d')
    return today_date_str + hour_str


def get_next_period_minute_window_date(minute_period, date):
    if date.minute % minute_period == 0:
        return date
    else:
        return date + pd.Timedelta(minutes=minute_period - date.minute % minute_period)


def get_leveraged_etfs():
    leveraged_tickers = [{
        '1x': 'SPY',
        '2x': 'SSO',
        '3X': 'SPXL'
    }, {
        '1x': 'QQQ',
        '2x': 'QLD',
        '3X': 'TQQQ'
    }, {
        '1x': 'IWM',
        '2x': 'UWM',
        '3X': 'URTY'
    }, {
        '1x': 'XLF',
        '3X': 'FAS'
    }, {
        '1x': 'XLE',
        '2x': 'ERX',
        '3X': 'OILU'
    }, {
        '1x': 'XLU',
        '3X': 'UTSL'
    }, {
        '1x': 'XLV',
        '3X': 'CURE'
    }, {
        '1x': 'XLI',
        '3x': 'DUSL'
    }, {
        '1x': 'XLP'
    }]
    return leveraged_tickers


def read_stock_from_file(ticker_name, dir_name):
    directory = os.fsencode(dir_name)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and filename == ticker_name + ".csv":
            df = pd.read_csv(dir_name + '/' + filename)
            return df
    return None