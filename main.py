import pandas as pd
import numpy as np
import seaborn as seaborn

from data_fetcher import get_sp500_list, get_data_dict_for_all_stocks_in_directory, get_data_dict_for_multiple_stocks, \
    get_data_for_stock, get_stock_data_trade_daily_alpha_vantage
from strategies import calculate_exits_column_by_atr_and_prev_max_min
from indicators import get_ma_column_for_stock, get_distance_between_columns_for_stock, \
    get_adx_column_for_stock, rsi, stochastic, get_ATR_column_for_stock, get_volatility_from_atr, \
    get_macd_columns_for_stock, normalize_columns, get_beta_column, get_breakout_column_for_stock, \
    get_touch_and_return_above_column_for_stock
from signals import indicators_mid_levels_signal, parabolic_trending_n_periods, cross_20_ma, cross_50_ma, joint_signal, \
    macd_cross_0_signal, macd_signal_cross_signal, joint_macd_signal_cross_signal, joint_macd_cross_0_signal, \
    awesome_oscilator, calculate_correl_score_series_for_df, cumulative_rsi_signal
import time
import matplotlib.pyplot as plt
import seaborn as sns
import datetime



# See PyCharm help at https://www.jetbrains.com/help/pycharm/

tickers = get_sp500_list()

adjusted_tickers = [elem for elem in tickers if elem != 'GOOG' and elem != 'DUK' and elem != 'HLT' and elem != 'DD' and elem != 'CMCSA' and elem != 'COG' and elem != 'WBA' and elem != 'KMX' and elem != 'ADP' and elem != 'STZ' and elem != 'IQV' and elem != 'BBWI' and elem != 'CTRA'] # there were stock splits
adjusted_tickers = [elem for elem in adjusted_tickers if '.' not in elem]
# yahoo finance screener - mega caps only, tech, energey and finance
adjusted_tickers = ['FB', 'AAPL', 'SPY']
# adjusted_tickers = ['AAPL']

# adjusted_tickers = adjusted_tickers + ['SPY', 'QQQ', 'IWM']
# adjusted_tickers = adjusted_tickers + ['SPY']

stocks_dict = get_data_dict_for_multiple_stocks(adjusted_tickers, time)
# spy_df = stocks_dict['SPY']

# stocks_dict, adjusted_tickers = get_data_dict_for_all_stocks_in_directory('stocks_csvs_raw')

# adjusted_tickers = ['FB', 'AAPL', 'NFLX', 'GOOGL', 'MSFT', 'AMZN', 'SPY', 'QQQ', 'IWM']

# stocks_dict = { tick: stocks_dict[tick].iloc[-(252*4):].reset_index(drop=True) for tick in adjusted_tickers }

adjusted_tickers_copy_1 = adjusted_tickers.copy()
for ticker in adjusted_tickers_copy_1:
    if ticker not in stocks_dict:
        adjusted_tickers.remove(ticker)


def split_df_to_train_test_sets(df, train_size_weeks, test_size_weeks):
    """
    :param train_size_weeks: number indicating number of weeks
    :param test_size_weeks: number indicating number of weeks
    :return: train_dfs list, test_dfs list
    """
    # check for datetime index, if not change it
    if not isinstance(df, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df['Date'], format ='%Y-%m-%d %H:%M:%S')
    one_day_df_group = df.groupby(pd.Grouper(key='Date', freq='W'))
    dfs = [one_day_df for _, one_day_df in one_day_df_group]

    resulting_train_dfs = []
    resulting_test_dfs = []
    while len(dfs) >= train_size_weeks:
        current_train_df_merged = pd.concat(dfs[:train_size_weeks])
        resulting_train_dfs.append(current_train_df_merged)
        resulting_test_dfs.extend(dfs[train_size_weeks:train_size_weeks + test_size_weeks])
        del dfs[:test_size_weeks]

    return resulting_train_dfs, resulting_test_dfs


def get_indicators_for_df(df):
    df['10_ma'] = get_ma_column_for_stock(df, 'Close', 10)
    df['20_ma'] = get_ma_column_for_stock(df, 'Close', 20)
    df['50_ma'] = get_ma_column_for_stock(df, 'Close', 50)
    df['200_ma'] = get_ma_column_for_stock(df, 'Close', 200)
    df['10_ma_volume'] = get_ma_column_for_stock(df, 'Volume', 10)
    df['20_ma_volume'] = get_ma_column_for_stock(df, 'Volume', 20)
    df['50_ma_volume'] = get_ma_column_for_stock(df, 'Volume', 50)
    df['10_ma_volume_break'] = get_breakout_column_for_stock(df, 'Volume', '10_ma_volume', '10_ma_volume_break')
    df['20_ma_volume_break'] = get_breakout_column_for_stock(df, 'Volume', '20_ma_volume', '20_ma_volume_break')
    df['50_ma_volume_break'] = get_breakout_column_for_stock(df, 'Volume', '50_ma_volume', '50_ma_volume_break')
    df['10_ma_touch'] = get_touch_and_return_above_column_for_stock(df, 'Close', '10_ma', '10_ma_touch', 4)
    df['20_ma_touch'] = get_touch_and_return_above_column_for_stock(df, 'Close', '20_ma', '20_ma_touch', 4)
    df['50_ma_touch'] = get_touch_and_return_above_column_for_stock(df, 'Close', '50_ma', '50_ma_touch', 4)
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
    df['distance_from_10_ma'] = get_distance_between_columns_for_stock(df, 'Close', '10_ma')
    df['adx'], df['+di'], df['-di'] = get_adx_column_for_stock(df, 14)
    df['adx_ma_med_5_rat'] = df['adx'] * df['ma_med_5_ratio']
    df['rsi'] = rsi(df, 14)  # changed from 14
    df['stochastic_k'], df['stochastic_d'] = stochastic(df, 14, 3)
    df['atr_volatility'], df['atr_volatility_ma'] = get_volatility_from_atr(df, 14)
    return df


def get_signals_for_df(df):
    df['signal_type'] = ''
    df['signal_direction'] = ''
    # signal_type and signal_direction columns are the columns that determine the actual orders!
    df = awesome_oscilator(df, 'signal_direction', 'signal_type')
    return df


def calculate_returns_for_df(df, n_prev_periods_check, ticker_name):
    df = calculate_exits_column_by_atr_and_prev_max_min(df, n_prev_periods_check, ticker_name)
    return df


def get_all_actions_df(df):
    actions_df = df.loc[df['in_position'] != ''].copy()
    return actions_df


def get_correls_on_norm_columns(df, cols):
    copied_df = df.copy()
    corr_dict = {}
    copied_df = copied_df.replace(r'^\s*$', np.NaN, regex=True)
    for col in cols:
        corr_dict[f'{col}_norm'] = copied_df['action_return_on_signal_index'].fillna(0).astype(float).corr(copied_df[f'{col}_norm'].fillna(0).astype(float))
    print(f'correlations with action return on signal index summary: {corr_dict}')

    # TODO: winning keys should be calculated on an automated monthly basis and pushed to a db, pulled in orders_notifier (taking into account preventing overfitting and less correlated features)
    # TODO: Update position scores in orders notifier!!!
    # winning_keys = ['10_ma_volume_norm', 'ma_med_34_ratio_norm', 'awesome_osc_norm', 'macd_signal_norm', 'distance_from_10_ma_norm']
    # winning_corr_dict = {winning_key: corr_dict[winning_key] for winning_key in winning_keys}
    # return winning_corr_dict
    return corr_dict


def append_df_in_list_check_index_exists(current_index, df_list, df_to_append):
    if current_index >= len(df_list):  # check if current_df_index does not exist on list. if so, add a df there
        df_list.append(pd.DataFrame())
    df_list[current_index] = pd.concat([df_list[current_index], df_to_append])
    return df_list[current_index]


def apply_features_for_dfs(dfs_list, with_actions=False):
    for current_index in range(len(dfs_list)):
        dfs_list[current_index] = get_indicators_for_df(dfs_list[current_index])
        dfs_list[current_index] = get_signals_for_df(dfs_list[current_index])
        if with_actions:
            dfs_list[current_index] = calculate_returns_for_df(dfs_list[current_index], 70, ticker)
        dfs_list[current_index] = dfs_list[current_index].reset_index()
    return dfs_list


def split_dfs_for_all_tickers(stocks_dict, tickers):
    splitted_stocks_dict = {}
    for ticker in tickers:
        print(f'splitting ticker: {ticker}')
        splitted_stocks_dict[ticker] = {}
        splitted_stocks_dict[ticker]['train_dfs'], splitted_stocks_dict[ticker]['test_dfs'] = split_df_to_train_test_sets(stocks_dict[ticker], 8, 1)
    return splitted_stocks_dict


def apply_features_for_splitted_stocks_dict(splitted_stocks_dict, tickers):
    for ticker in tickers:
        print(f'applying features for splitted ticker: {ticker}')
        splitted_stocks_dict[ticker]['train_dfs'] = apply_features_for_dfs(splitted_stocks_dict[ticker]['train_dfs'], with_actions=True)
        splitted_stocks_dict[ticker]['test_dfs'] = apply_features_for_dfs(splitted_stocks_dict[ticker]['test_dfs'], with_actions=False)
    return splitted_stocks_dict


def combine_dfs_for_all_stocks_by_index(splitted_stocks_dict, data_group, tickers):
    # data_group = 'train_dfs' / 'test_dfs'
    all_data_group_dfs = []
    for ticker in tickers:
        for current_index in range(len(splitted_stocks_dict[ticker][data_group])):
            all_data_group_dfs[current_index] = append_df_in_list_check_index_exists(current_index, all_data_group_dfs, splitted_stocks_dict[ticker][data_group][current_index])
            all_data_group_dfs[current_index].loc[:, 'ticker'] = ticker
    return all_data_group_dfs


def get_all_actions_dfs_list(df_list):
    actions_dfs = []
    for current_index in range(len(df_list)):
        actions_dfs[current_index] = get_all_actions_df(df_list[current_index])
    return actions_dfs


def get_only_entrances_dfs_list(df_list):
    entrances_dfs = []
    for current_index in range(len(df_list)):
        entrances_dfs[current_index] = df_list[current_index].copy()[df_list[current_index]['action_return_on_signal_index'] != '']
    return entrances_dfs


def normalize_dfs(df_list, columns_to_normalize):
    normalized_dfs = []
    for current_index in range(len(df_list)):
        normalized_dfs[current_index] = normalize_columns(df_list[current_index], columns_to_normalize)
    return normalized_dfs


def get_correls_dicts_for_train_dfs(df_list, columns_to_normalize):
    correls_dicts = []
    for current_index in range(len(df_list)):
        correls_dicts[current_index] = get_correls_on_norm_columns(df_list[current_index], columns_to_normalize)
    return correls_dicts


def calculate_correl_score_series_for_dfs(df_list, correls_dicts_list):
    dfs_with_scores = []
    for current_index in range(len(df_list)):
        dfs_with_scores[current_index] = calculate_correl_score_series_for_df(df_list[current_index], correls_dicts_list[current_index])
    return dfs_with_scores


all_splitted_stocks_dict = split_dfs_for_all_tickers(stocks_dict, adjusted_tickers)
all_splitted_stocks_dict = apply_features_for_splitted_stocks_dict(all_splitted_stocks_dict, adjusted_tickers)

combined_train_dfs_for_all_stocks_by_index = combine_dfs_for_all_stocks_by_index(all_splitted_stocks_dict, 'train_dfs', adjusted_tickers)
combined_test_dfs_for_all_stocks_by_index = combine_dfs_for_all_stocks_by_index(all_splitted_stocks_dict, 'test_dfs', adjusted_tickers)
all_actions_train_dfs = get_all_actions_dfs_list(combined_train_dfs_for_all_stocks_by_index)
only_entrances_train_dfs = get_all_actions_dfs_list(all_actions_train_dfs)
columns_to_normalize = ['Volume', '10_ma_volume', '20_ma_volume', '50_ma_volume',
                            # '10_beta_SPY', '50_beta_SPY',
                            'median_ratio', 'ma_med_5_ratio',
                            'ma_med_34_ratio', 'awesome_osc', 'macd', 'macd_signal',
                            'distance_from_10_ma', 'adx', '+di', '-di', 'rsi', 'stochastic_k',
                            'stochastic_d', 'atr_volatility', 'atr_volatility_ma']
normalized_train_dfs = normalize_dfs(only_entrances_train_dfs, columns_to_normalize)
correls_dicts_for_train_dfs = get_correls_dicts_for_train_dfs(normalized_train_dfs, columns_to_normalize)
normalized_train_dfs_with_scores = calculate_correl_score_series_for_dfs(normalized_train_dfs, correls_dicts_for_train_dfs)



# start = time.time()
# all_actions_dfs_train = []
# correls_dicts_from_train_dfs = []
# only_entrances_dfs_train = []
# all_actions_dfs_test = []
# for ticker in adjusted_tickers:
#     print(f'engineering ticker: {ticker}')
#     train_dfs, test_dfs = split_df_to_train_test_sets(stocks_dict[ticker], 8, 1)
#     for current_train_test_index in range(len(train_dfs)):
#         train_dfs[current_train_test_index] = get_indicators_for_df(train_dfs[current_train_test_index])
#         train_dfs[current_train_test_index] = get_signals_for_df(train_dfs[current_train_test_index])
#         train_dfs[current_train_test_index] = calculate_returns_for_df(train_dfs[current_train_test_index], 70, ticker)
#         train_dfs[current_train_test_index] = train_dfs[current_train_test_index].reset_index()
#         all_actions_df_for_ticker_for_current_df_index = get_all_actions_df(train_dfs[current_train_test_index])
#
#         all_actions_dfs_train[current_train_test_index] = append_df_in_list_check_index_exists(current_train_test_index, all_actions_dfs_train, all_actions_df_for_ticker_for_current_df_index)
#
#         columns_to_normalize = ['Volume', '10_ma_volume', '20_ma_volume', '50_ma_volume',
#                                 # '10_beta_SPY', '50_beta_SPY',
#                                 'median_ratio', 'ma_med_5_ratio',
#                                 'ma_med_34_ratio', 'awesome_osc', 'macd', 'macd_signal',
#                                 'distance_from_10_ma', 'adx', '+di', '-di', 'rsi', 'stochastic_k',
#                                 'stochastic_d', 'atr_volatility', 'atr_volatility_ma']
#         # TODO: I should have another function that drops correls that are highly correlated automatically but leave the one of them that has the highest correlation with the label
#         all_actions_dfs_train[current_train_test_index] = normalize_columns(all_actions_dfs_train[current_train_test_index], columns_to_normalize)
#         only_entrances_df_train = all_actions_dfs_train[current_train_test_index].copy()[all_actions_dfs_train[current_train_test_index]['action_return_on_signal_index'] != '']
#         correls_dict_train_for_index = get_correls_on_norm_columns(only_entrances_df_train, columns_to_normalize)
#
#         # TODO: This is not correct. This should be a single correls dict for all stocks in the index. from that draw conclusion for all test stocks in the same index
#         if current_train_test_index >= len(correls_dicts_from_train_dfs): # check if current_df_index does not exist on all_actions_dfs_train. this will have to be done on the tests as well
#             correls_dicts_from_train_dfs.append({})
#         # TODO: Now what should I do here? its not a df so shouldn't be using pandas. Can't use averages since its wrong in correlations. I should only be doing this once I have all the tickers for each index!
#         correls_dicts_from_train_dfs[current_train_test_index] = pd.concat([correls_dicts_from_train_dfs[current_train_test_index], correls_dict_train_for_index])
#
#
#         only_entrances_dfs_train = calculate_correl_score_series_for_df(only_entrances_df_train, correls_dict_train_for_index)
#
#         # TEST DF Engineering
#
#         test_dfs[current_test_df_index] = get_indicators_for_df(test_dfs[current_test_df_index])
#         test_dfs[current_test_df_index] = get_signals_for_df(test_dfs[current_test_df_index])
#         # TODO: add scoring calculation for test df out of current train df cors results
#         # TODO: calculate returns based on scoring as well. Somehow edit the below function for dealing with scoring.
#         # TODO: or have a function that calculates returns with scoring and another function that calculates returns without scoring. use the without one for train and the with one for test
#         test_dfs[current_test_df_index] = calculate_returns_for_df(test_dfs[current_test_df_index], 70, ticker)
#         test_dfs[current_test_df_index] = test_dfs[current_test_df_index].reset_index()
#         all_actions_df_for_ticker_for_current_df_index = get_all_actions_df(train_dfs[current_test_df_index])
#
#         if current_test_df_index >= len(all_actions_dfs_train): # check if current_df_index does not exist on all_actions_dfs_train. this will have to be done on the tests as well
#             all_actions_dfs_test.append(pd.DataFrame())
#         all_actions_dfs_test[current_test_df_index] = pd.concat([all_actions_dfs_test[current_test_df_index], all_actions_df_for_ticker_for_current_df_index])
#
#     # TODO: still need to figure out if, where and how im saving this shit
#     stocks_dict[ticker].to_csv(f'stocks_csvs_new/{ticker}_engineered.csv', index=False)
#
# end = time.time()
# print(f'time for processing all stocks: {end - start}')



