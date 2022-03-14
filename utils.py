import os
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