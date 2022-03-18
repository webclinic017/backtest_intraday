import math
import os
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as seaborn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

from joblib import dump, load

from backtest_intraday import backtest_intraday
from data_fetcher import get_sp500_list, get_data_dict_for_all_stocks_in_directory, get_data_dict_for_multiple_stocks, \
    get_data_for_stock, get_stock_data_trade_daily_alpha_vantage, get_dfs_for_all_csvs_in_directory
from strategies import calculate_exits_column_by_atr_and_prev_max_min, calculate_returns_for_df_based_on_signals_alone
from indicators import get_ma_column_for_stock, get_distance_between_columns_for_stock, \
    get_adx_column_for_stock, rsi, stochastic, get_ATR_column_for_stock, get_volatility_from_atr, \
    get_macd_columns_for_stock, normalize_columns, get_beta_column, get_breakout_column_for_stock, \
    get_touch_and_return_above_column_for_stock, normalize_columns_with_predefined_scaler, slope, simple_slope
from signals import indicators_mid_levels_signal, parabolic_trending_n_periods, cross_20_ma, cross_50_ma, joint_signal, \
    macd_cross_0_signal, macd_signal_cross_signal, joint_macd_signal_cross_signal, joint_macd_cross_0_signal, \
    awesome_oscilator, calculate_correl_score_series_for_df, cumulative_rsi_signal, crossing_mas
import time
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from utils import is_business_day, is_early_close_business_day


print(f'{datetime.datetime.now()} : starting main...')

scheduler = BlockingScheduler()


def scheduled_backtest_job():
    print(f'date: {datetime.datetime.now()}, start running job')
    backtest_intraday()
    print(f'date: {datetime.datetime.now()}, finished running job')

scheduler.add_job(scheduled_backtest_job, 'cron', day_of_week='sun', hour='10', minute='10', timezone='US/Eastern')

# TODO: there is a calendar api in alpaca that can reveal if there is trading today and when it starts and when it ends...
# TODO: ...I can maybe use it to define when to run cron job

scheduler.start()
