import os
import pandas as pd
import numpy as np
import seaborn as seaborn

from data_fetcher import get_sp500_list, get_data_dict_for_all_stocks_in_directory, get_data_dict_for_multiple_stocks, \
    get_data_for_stock, get_stock_data_trade_daily_alpha_vantage
from strategies import calculate_exits_column_by_atr_and_prev_max_min
from indicators import get_ma_column_for_stock, get_distance_between_columns_for_stock, \
    get_adx_column_for_stock, rsi, stochastic, get_ATR_column_for_stock, get_volatility_from_atr, \
    get_macd_columns_for_stock, normalize_columns, get_beta_column, get_breakout_column_for_stock, \
    get_touch_and_return_above_column_for_stock, normalize_columns_with_predefined_scaler
from signals import indicators_mid_levels_signal, parabolic_trending_n_periods, cross_20_ma, cross_50_ma, joint_signal, \
    macd_cross_0_signal, macd_signal_cross_signal, joint_macd_signal_cross_signal, joint_macd_cross_0_signal, \
    awesome_oscilator, calculate_correl_score_series_for_df, cumulative_rsi_signal
import time
import matplotlib.pyplot as plt
import seaborn as sns
import datetime



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
from utils import save_create_csv

# tickers = get_sp500_list()
# adjusted_tickers = [elem for elem in tickers if elem != 'GOOG' and elem != 'DUK' and elem != 'HLT' and elem != 'DD' and elem != 'CMCSA' and elem != 'COG' and elem != 'WBA' and elem != 'KMX' and elem != 'ADP' and elem != 'STZ' and elem != 'IQV' and elem != 'BBWI' and elem != 'CTRA'] # there were stock splits
# adjusted_tickers = [elem for elem in adjusted_tickers if '.' not in elem]

# adjusted_tickers = ['FB', 'AAPL', 'SPY', 'IWM', 'QQQ', 'AMZN', 'TSLA', 'GOOGL', 'AAL', 'WYNN', 'MMM', 'DIS', 'NFLX', 'AMD', 'INTL', 'MS', 'IVZ', 'AZO', 'IT', 'T', 'VZ', 'QCOM', 'MGM', 'BLK', 'NVDA', 'PYPL', 'MRNA', 'TEVA', 'XLF', 'XLE', 'XLU', 'JPM', 'V', 'BAC', 'TSM', 'JNJ', 'WMT']
adjusted_tickers = ['SPY', 'IWM', 'QQQ', 'XLF', 'XLE', 'XLU']

# adjusted_tickers = adjusted_tickers + ['SPY', 'QQQ', 'IWM']

# stocks_dict = get_data_dict_for_multiple_stocks(adjusted_tickers, time)

stocks_dict, adjusted_tickers = get_data_dict_for_all_stocks_in_directory('stocks_csvs_raw')

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
    one_week_df_group = df.groupby(pd.Grouper(key='Date', freq='W'))
    dfs = [one_week_df for _, one_week_df in one_week_df_group]

    resulting_train_dfs = []
    resulting_test_dfs = []
    while len(dfs) > train_size_weeks:
        current_train_df_merged = pd.concat(dfs[:train_size_weeks])
        resulting_train_dfs.append(current_train_df_merged)
        current_test_df_merged = pd.concat(dfs[train_size_weeks:train_size_weeks + test_size_weeks])
        resulting_test_dfs.append(current_test_df_merged)
        del dfs[:test_size_weeks]

    return resulting_train_dfs, resulting_test_dfs


def get_indicators_for_df(df, ticker):
    print(f'getting indicators for {ticker}')
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


def get_signals_for_df(df, ticker):
    print(f'getting signals for {ticker}')
    df['signal_type'] = None
    df['signal_direction'] = None
    # signal_type and signal_direction columns are the columns that determine the actual orders!
    df = awesome_oscilator(df, 'signal_direction', 'signal_type')
    return df


def calculate_returns_for_df(df, n_prev_periods_check, ticker_name):
    # TODO: toggle comment between these 2 lines for daily/intraday data
    df = calculate_exits_column_by_atr_and_prev_max_min(df, n_prev_periods_check, ticker_name, 'intraday')
    # df = calculate_exits_column_by_atr_and_prev_max_min(df, n_prev_periods_check, ticker_name, 'daily')
    return df


def get_all_actions_df(df):
    actions_df = df.loc[df['in_position'] == True]
    return actions_df


def get_correls_on_norm_columns(df, cols):
    if df.empty or len(df) < 5:
        return {}
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
    df_list[current_index] = pd.concat([df_list[current_index], df_to_append.copy()])
    return df_list[current_index]


def apply_features_for_dfs(dfs_list, ticker, data_group, with_actions=False):
    # data_group = 'train_dfs' / 'test_dfs'
    for current_index in range(len(dfs_list)):
        print(f'applying features for {ticker}, current df index {current_index}, with actions {with_actions}')
        dfs_list[current_index].index.name = 'original_index'
        dfs_list[current_index] = dfs_list[current_index].reset_index()
        dfs_list[current_index] = get_indicators_for_df(dfs_list[current_index], ticker)
        dfs_list[current_index] = get_signals_for_df(dfs_list[current_index], ticker)
        if with_actions:
            dfs_list[current_index] = calculate_returns_for_df(dfs_list[current_index], 70, ticker)
        if data_group == 'train_dfs':
            save_create_csv('stocks_csvs_splits', f'{ticker}_train_{current_index}',  dfs_list[current_index])
        elif data_group == 'test_dfs':
            save_create_csv('stocks_csvs_splits', f'{ticker}_test_{current_index}',  dfs_list[current_index])
    return dfs_list


def apply_actions_for_dfs(dfs_list, ticker, data_group):
    # data_group = 'train_dfs' / 'test_dfs'
    for current_index in range(len(dfs_list)):
        print(f'applying actions for {ticker}, current df index {current_index}')
        dfs_list[current_index].index.name = 'original_index'
        dfs_list[current_index] = dfs_list[current_index].reset_index()
        dfs_list[current_index] = calculate_returns_for_df(dfs_list[current_index], 70, ticker)
        if data_group == 'train_dfs':
            save_create_csv('stocks_csvs_splits', f'{ticker}_train_{current_index}',  dfs_list[current_index])
        elif data_group == 'test_dfs':
            save_create_csv('stocks_csvs_splits', f'{ticker}_test_{current_index}',  dfs_list[current_index])
    return dfs_list


def split_dfs_for_all_tickers(stocks_dict, tickers):
    splitted_stocks_dict = {}
    for ticker in tickers:
        print(f'splitting ticker: {ticker}')
        splitted_stocks_dict[ticker] = {}
        splitted_stocks_dict[ticker]['train_dfs'], splitted_stocks_dict[ticker]['test_dfs'] = split_df_to_train_test_sets(stocks_dict[ticker], 8, 1)
        # splitted_stocks_dict[ticker]['train_dfs'], splitted_stocks_dict[ticker]['test_dfs'] = split_df_to_train_test_sets(stocks_dict[ticker], 52, 12)
    return splitted_stocks_dict


def apply_features_for_splitted_stocks_dict(splitted_stocks_dict, tickers):
    for ticker in tickers:
        print(f'applying features for splitted ticker: {ticker}')
        splitted_stocks_dict[ticker]['train_dfs'] = apply_features_for_dfs(splitted_stocks_dict[ticker]['train_dfs'], ticker, data_group='train_dfs', with_actions=True)
        splitted_stocks_dict[ticker]['test_dfs'] = apply_features_for_dfs(splitted_stocks_dict[ticker]['test_dfs'], ticker, data_group='test_dfs', with_actions=True)
    return splitted_stocks_dict


def apply_actions_for_splitted_stocks_dict(splitted_stocks_dict, tickers):
    for ticker in tickers:
        print(f'applying features for splitted ticker: {ticker}')
        splitted_stocks_dict[ticker]['train_dfs'] = apply_actions_for_dfs(splitted_stocks_dict[ticker]['train_dfs'], ticker, data_group='train_dfs')
        splitted_stocks_dict[ticker]['test_dfs'] = apply_actions_for_dfs(splitted_stocks_dict[ticker]['test_dfs'], ticker, data_group='test_dfs')
    return splitted_stocks_dict


def combine_dfs_for_all_stocks_by_index(splitted_stocks_dict, data_group, tickers):
    # data_group = 'train_dfs' / 'test_dfs'
    all_data_group_dfs = []
    for ticker in tickers:
        for current_index in range(len(splitted_stocks_dict[ticker][data_group])):
            all_data_group_dfs[current_index] = append_df_in_list_check_index_exists(current_index, all_data_group_dfs, splitted_stocks_dict[ticker][data_group][current_index])
            all_data_group_dfs[current_index] = all_data_group_dfs[current_index].reset_index(drop=True)
    return all_data_group_dfs


def get_all_actions_dfs_list(df_list):
    actions_dfs = []
    print(f'getting all actions dfs')
    for current_index in range(len(df_list)):
        print(f'current index {current_index}')
        actions_dfs.append(get_all_actions_df(df_list[current_index]))
    return actions_dfs


def get_only_entrances_dfs_list(df_list):
    entrances_dfs = []
    print(f'getting all entrances dfs')
    for current_index in range(len(df_list)):
        print(f'current index {current_index}')
        entrances_dfs.append(df_list[current_index].copy()[df_list[current_index]['action_return_on_signal_index'].notnull()])
    return entrances_dfs


def normalize_dfs(df_list, columns_to_normalize):
    normalized_dfs = []
    scalers = []
    print(f'getting all normalized dfs')
    for current_index in range(len(df_list)):
        print(f'current index {current_index}')
        normalized_df, current_scalers = normalize_columns(df_list[current_index], columns_to_normalize)
        normalized_dfs.append(normalized_df)
        scalers.append(current_scalers)
    return normalized_dfs, scalers


def normalize_dfs_with_predefined_scalers(df_list, columns_to_normalize, scalers):
    normalized_dfs = []
    print(f'getting all normalized dfs')
    for current_index in range(len(df_list)):
        print(f'current index {current_index}')
        normalized_df = normalize_columns_with_predefined_scaler(df_list[current_index], columns_to_normalize, scalers[current_index])
        normalized_dfs.append(normalized_df)
    return normalized_dfs


def get_all_correls_df(correls_dicts):
    all_correls_dict = {}
    for dicts_index in range(len(correls_dicts)):
        for key, value in correls_dicts[dicts_index].items():
            if key not in all_correls_dict:
                all_correls_dict[key] = []
            all_correls_dict[key].append(correls_dicts[dicts_index][key])
    all_correls_df = pd.DataFrame.from_dict(all_correls_dict)
    save_create_csv('train_correls', 'all_train_correls', all_correls_df.reset_index())
    return all_correls_df


def get_correls_dicts_for_train_dfs(df_list, columns_to_normalize):
    correls_dicts = []
    print(f'getting all correls dicts')
    for current_index in range(len(df_list)):
        print(f'current index {current_index}')
        correls_dicts.append(get_correls_on_norm_columns(df_list[current_index], columns_to_normalize))
    return correls_dicts


def get_df_without_norm_columns(df):
    df_columns_without_norms = [c for c in df.columns if 'norm' not in c]
    return df[df_columns_without_norms]


def calculate_correl_score_series_for_dfs(df_list, best_correls_df, data_group):
    # data_group could be 'train'/'test'
    dfs_with_scores = []
    print(f'getting all correls scores for dfs')
    for current_index in range(len(df_list)):
        print(f'current index {current_index}')
        if df_list[current_index].empty:
            dfs_with_scores.append(df_list[current_index])
            continue
        current_df_with_scores = calculate_correl_score_series_for_df(df_list[current_index], best_correls_df.to_dict('records')[current_index])
        current_df_with_scores['df_index'] = current_index
        if data_group == 'train':
            save_create_csv('train_dfs_with_scores', f'train_df_{current_index}', get_df_without_norm_columns(current_df_with_scores))
        # elif data_group == 'test':
        #     save_create_csv('test_dfs_with_scores', f'test_df_{current_index}', get_df_without_norm_columns(current_df_with_scores))
        dfs_with_scores.append(current_df_with_scores)
    return dfs_with_scores


def get_best_correls_df(correls_df):
    correls_df_copy = correls_df.copy()
    std_series = correls_df_copy.std()
    median_std = std_series.median()
    column_names_below_median_std = list(std_series.loc[std_series <= median_std].index.values)
    correls_abs_df = correls_df_copy[column_names_below_median_std].abs()
    abs_avg_series = correls_abs_df.mean()
    median_abs_avg = abs_avg_series.median()
    column_names_above_avg_abs = list(abs_avg_series.loc[abs_avg_series > median_abs_avg].index.values)
    print(abs_avg_series.loc[abs_avg_series > median_abs_avg])
    # TODO: clear highly correlated columns and run through this process again to replace the bad columns
    return correls_df[column_names_above_avg_abs]


def read_splitted_stocks_dfs():
    print('reading splitted stocks from files')
    splitted_stocks_dict = {}
    directory = os.fsencode('stocks_csvs_splits')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(f'handling {filename}')
        if filename.endswith(".csv"):
            ticker = filename.split('_')[0]
            if ticker not in splitted_stocks_dict:
                splitted_stocks_dict[ticker] = { 'train_dfs': [], 'test_dfs': [] }
            data_group = filename.split('_')[1]
            stock_df = pd.read_csv('stocks_csvs_splits/' + filename)
            if data_group == 'train':
                splitted_stocks_dict[ticker]['train_dfs'].append(stock_df)
            elif data_group == 'test':
                splitted_stocks_dict[ticker]['test_dfs'].append(stock_df)
    return splitted_stocks_dict


def apply_features_for_stocks(all_stocks_dict, tickers):
    for ticker in tickers:
        print(f'applying features for full stock {ticker}')
        all_stocks_dict[ticker] = get_indicators_for_df(all_stocks_dict[ticker], ticker)
        all_stocks_dict[ticker] = get_signals_for_df(all_stocks_dict[ticker], ticker)
        save_create_csv('full_stocks_csvs_with_features', f'{ticker}',  all_stocks_dict[ticker])
    return all_stocks_dict


# TODO: There is a problem here with short length dfs - many lengthy features wont exist. so need to first apply the features and then split. FIXED
# all_splitted_stocks_dict = split_dfs_for_all_tickers(stocks_dict, adjusted_tickers)
# all_splitted_stocks_dict = apply_features_for_splitted_stocks_dict(all_splitted_stocks_dict, adjusted_tickers)

# all_splitted_stocks_dict = read_splitted_stocks_dfs()

all_stocks_dict_with_features = apply_features_for_stocks(stocks_dict, adjusted_tickers)
all_stocks_dict_with_features_splitted = split_dfs_for_all_tickers(all_stocks_dict_with_features, adjusted_tickers)
all_splitted_stocks_dict = apply_actions_for_splitted_stocks_dict(all_stocks_dict_with_features_splitted, adjusted_tickers)

combined_train_dfs_for_all_stocks_by_index = combine_dfs_for_all_stocks_by_index(all_splitted_stocks_dict, 'train_dfs', adjusted_tickers)
combined_test_dfs_for_all_stocks_by_index = combine_dfs_for_all_stocks_by_index(all_splitted_stocks_dict, 'test_dfs', adjusted_tickers)
columns_to_normalize = ['Volume', '10_ma_volume', '20_ma_volume', '50_ma_volume',
                            # '10_beta_SPY', '50_beta_SPY',
                            'median_ratio', 'ma_med_5_ratio',
                            'ma_med_34_ratio', 'awesome_osc', 'macd', 'macd_signal',
                            'distance_from_10_ma', 'adx', '+di', '-di', 'rsi', 'stochastic_k',
                            'stochastic_d', 'atr_volatility', 'atr_volatility_ma']
normalized_train_dfs, train_scalers = normalize_dfs(combined_train_dfs_for_all_stocks_by_index, columns_to_normalize)
all_actions_train_dfs = get_all_actions_dfs_list(normalized_train_dfs)
only_entrances_train_dfs = get_only_entrances_dfs_list(all_actions_train_dfs)
correls_dicts_for_train_dfs = get_correls_dicts_for_train_dfs(only_entrances_train_dfs, columns_to_normalize)
all_train_correls_df = get_all_correls_df(correls_dicts_for_train_dfs)
best_train_correls_df = get_best_correls_df(all_train_correls_df)
normalized_train_dfs_with_scores = calculate_correl_score_series_for_dfs(only_entrances_train_dfs, best_train_correls_df, 'train')

all_trains_df_with_scores = pd.concat(normalized_train_dfs_with_scores)
save_create_csv('train_dfs_with_scores', 'all_trains_df_with_scores', all_trains_df_with_scores)

all_actions_test_dfs = get_all_actions_dfs_list(combined_test_dfs_for_all_stocks_by_index)
normalized_test_dfs = normalize_dfs_with_predefined_scalers(all_actions_test_dfs, columns_to_normalize, train_scalers)
only_entrances_test_dfs = get_only_entrances_dfs_list(normalized_test_dfs)
normalized_test_dfs_with_scores = calculate_correl_score_series_for_dfs(only_entrances_test_dfs, best_train_correls_df, 'test')

for current_index in range(len(normalized_test_dfs_with_scores)):
    save_create_csv('test_dfs_with_scores', f'test_df_{current_index}',
                    get_df_without_norm_columns(normalized_test_dfs_with_scores[current_index]))
all_test_df_with_scores = pd.concat(normalized_test_dfs_with_scores)
save_create_csv('test_dfs_with_scores', 'all_test_df_with_scores', all_test_df_with_scores)
