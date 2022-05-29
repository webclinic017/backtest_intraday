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
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from statsmodels.stats.outliers_influence import variance_inflation_factor

from joblib import dump, load

from data_fetcher import get_sp500_list, get_data_dict_for_all_stocks_in_directory, get_data_dict_for_multiple_stocks, \
    get_data_for_stock, get_stock_data_trade_daily_alpha_vantage, get_dfs_for_all_csvs_in_directory
from stock_utils import get_only_trading_hours_from_df_dict, apply_features_for_stocks, get_indicators_for_df, \
    get_signals_for_df
from strategies import calculate_exits_column_by_atr_and_prev_max_min, calculate_returns_for_df_based_on_signals_alone
from indicators import get_ma_column_for_stock, get_distance_between_columns_for_stock, \
    get_adx_column_for_stock, rsi, stochastic, get_ATR_column_for_stock, get_volatility_from_atr, \
    get_macd_columns_for_stock, normalize_columns, get_beta_column, get_breakout_column_for_stock, \
    get_touch_and_return_above_column_for_stock, normalize_columns_with_predefined_scaler, slope, simple_slope, \
    columns_to_normalize
from signals import indicators_mid_levels_signal, parabolic_trending_n_periods, cross_20_ma, cross_50_ma, joint_signal, \
    macd_cross_0_signal, macd_signal_cross_signal, joint_macd_signal_cross_signal, joint_macd_cross_0_signal, \
    awesome_oscilator, calculate_correl_score_series_for_df, cumulative_rsi_signal, crossing_mas
import time
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
from utils import save_create_csv, get_feature_col_names


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
        corr_dict[f'{col}_norm'] = copied_df['price_change_for_corr'].fillna(0).astype(float).corr(copied_df[f'{col}_norm'].fillna(0).astype(float))
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
            # TODO: depending on the type of strategy I want, toggle comments
            # dfs_list[current_index] = calculate_returns_for_df(dfs_list[current_index], 70, ticker)
            dfs_list[current_index] = calculate_returns_for_df_based_on_signals_alone(dfs_list[current_index], ticker)
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
        # TODO: depending on the type of strategy I want, toggle comments
        # dfs_list[current_index] = calculate_returns_for_df(dfs_list[current_index], 70, ticker)
        dfs_list[current_index] = calculate_returns_for_df_based_on_signals_alone(dfs_list[current_index], ticker)
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


def transformation_sin_cos(column):
  max_value = column.max()
  sin_values = [math.sin((2 * math.pi * x) / max_value) for x in list(column)]
  cos_values = [math.cos((2 * math.pi * x) / max_value) for x in list(column)]
  return sin_values, cos_values


def construct_categorical_cols_for_dfs(df_list, all_tickers):
    ticker_map = { index: ticker_name for ticker_name, index in enumerate(all_tickers) }
    for i in range(len(df_list)):
        df_list[i]['ticker'] = df_list[i]['ticker'].map(ticker_map)
        df_list[i]['ticker_sin'], df_list[i]['ticker_cos'] = transformation_sin_cos(df_list[i]['ticker'])
        df_list[i]['day_of_week'] = pd.to_datetime(df_list[i]['Date'], format ='%Y-%m-%d %H:%M:%S').dt.dayofweek
        df_list[i]['day_of_week_sin'], df_list[i]['day_of_week_cos'] = transformation_sin_cos(df_list[i]['day_of_week'])
        df_list[i]['time_of_day'] = pd.to_datetime(df_list[i]['Date'], format ='%Y-%m-%d %H:%M:%S').dt.hour.apply(lambda x: 1 if x >= 13 else 0)
        df_list[i]['binary_signal'] = df_list[i]['signal'].map({ 'Bullish': 1, 'Bearish': 0 })
        df_list[i]['action_return_on_signal_index_categorical'] = df_list[i]['action_return_on_signal_index'] > 0.003
        df_list[i]['action_return_on_signal_index_categorical'] = df_list[i]['action_return_on_signal_index_categorical'].astype(int)
        df_list[i]['binary_5_ma_vol_break'] = df_list[i]['5_ma_volume_break'].astype(int)
        df_list[i]['binary_5_ma_touch'] = df_list[i]['5_ma_touch'].astype(int)
    return df_list


def drop_rows_with_na_values_from_dfs(df_list, cols_to_consider):
    # TODO: make sure this function does not mutate the original dfs in df_list
    for i in range(len(df_list)):
        current_df = df_list[i]
        df_list[i] = current_df.dropna(subset=cols_to_consider)
    return df_list


# def spot_check_algorithms(n_splits, features, labels):
#     models = []
#     models.append(('LR', LinearRegression()))
#     models.append(('NN', MLPRegressor(solver = 'lbfgs', max_iter=500)))
#     models.append(('KNN', KNeighborsRegressor()))
#     models.append(('RF', RandomForestRegressor(n_estimators = 10)))
#
#     # Evaluate each model in turn
#     results = []
#     names = []
#     for name, model in models:
#         tscv = TimeSeriesSplit(n_splits=n_splits)
#         cv_results = cross_val_score(model, features, labels.values.ravel(), cv=tscv, scoring='neg_mean_absolute_error')
#         results.append(cv_results)
#         names.append(name)
#         print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
#
#     # Compare Algorithms
#     plt.boxplot(results, labels=names)
#     plt.title('Algorithm Comparison')
#     plt.show()


def select_best_actions_using_machine_learning(train_dfs, test_dfs, feature_col_names, label_col_name):
    # the model will train on each train set and will predict on a corresponding test set.
    # we will predict future values according to the model resulted from all train sets.
    gpc = GaussianProcessClassifier(1*RBF(), random_state=0)
    # gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    over = SMOTE(sampling_strategy=0.3)
    # under = RandomUnderSampler(sampling_strategy=0.9)
    # steps = [('o', over), ('u', under)]
    steps = [('o', over)]
    pipeline = Pipeline(steps=steps)

    for i in range(len(train_dfs)):
        print(f'current index for model fit: {i}')
        train_features_array = train_dfs[i][feature_col_names].values
        train_label_array = train_dfs[i][label_col_name].values

        vif_data = pd.DataFrame()
        vif_data["feature"] = train_dfs[i][feature_col_names].columns
        # calculating VIF for each feature
        vif_data["VIF"] = [variance_inflation_factor(train_features_array, i)
                           for i in range(len(train_dfs[i][feature_col_names].columns))]
        print(f'vif_data: {vif_data}')

        test_features_array = test_dfs[i][feature_col_names].values
        test_label_array = test_dfs[i][label_col_name].values
        train_features_array, train_label_array = pipeline.fit_resample(train_features_array, train_label_array)
        counter = Counter(train_label_array)
        print(f'classes count: {counter}')
        gpc.fit(train_features_array, train_label_array)
        # print(gpc.predict_proba(test_features_array))
        print(gpc.score(test_features_array, test_label_array))
        test_dfs[i]['prediction'] = gpc.predict(test_features_array)
    dump(gpc, 'intraday_model.joblib') # TODO: in the realtime single day job, I should use model = load('intraday_model.joblib') to get the model
    return test_dfs



def backtest_intraday(adjusted_tickers):
    # tickers = get_sp500_list()
    # adjusted_tickers = [elem for elem in tickers if elem != 'GOOG' and elem != 'DUK' and elem != 'HLT' and elem != 'DD' and elem != 'CMCSA' and elem != 'COG' and elem != 'WBA' and elem != 'KMX' and elem != 'ADP' and elem != 'STZ' and elem != 'IQV' and elem != 'BBWI' and elem != 'CTRA'] # there were stock splits
    # adjusted_tickers = [elem for elem in adjusted_tickers if '.' not in elem]

    # adjusted_tickers = ['FB', 'AAPL', 'SPY', 'IWM', 'QQQ', 'AMZN', 'TSLA', 'GOOGL', 'AAL', 'WYNN', 'MMM', 'DIS', 'NFLX', 'AMD', 'INTL', 'MS', 'IVZ', 'AZO', 'IT', 'T', 'VZ', 'QCOM', 'MGM', 'BLK', 'NVDA', 'PYPL', 'MRNA', 'TEVA', 'XLF', 'XLE', 'XLU', 'JPM', 'V', 'BAC', 'TSM', 'JNJ', 'WMT']
    # adjusted_tickers = ['SPY', 'IWM', 'QQQ', 'XLF', 'XLE', 'XLU', 'XLV', 'XLI', 'XLP']
    # adjusted_tickers = ['SPY']

    # adjusted_tickers = adjusted_tickers + ['SPY', 'QQQ', 'IWM']

    stocks_dict = get_data_dict_for_multiple_stocks(adjusted_tickers, time)

    # stocks_dict, adjusted_tickers = get_data_dict_for_all_stocks_in_directory('stocks_csvs_raw')

    # stocks_dict = { tick: stocks_dict[tick].iloc[-(252*4):].reset_index(drop=True) for tick in adjusted_tickers }

    adjusted_tickers_copy_1 = adjusted_tickers.copy()
    for ticker in adjusted_tickers_copy_1:
        if ticker not in stocks_dict:
            adjusted_tickers.remove(ticker)

    stocks_dict = get_only_trading_hours_from_df_dict(stocks_dict, adjusted_tickers)
    all_stocks_dict_with_features = apply_features_for_stocks(stocks_dict, adjusted_tickers)
    all_stocks_dict_with_features_splitted = split_dfs_for_all_tickers(all_stocks_dict_with_features, adjusted_tickers)
    all_splitted_stocks_dict = apply_actions_for_splitted_stocks_dict(all_stocks_dict_with_features_splitted, adjusted_tickers)

    combined_train_dfs_for_all_stocks_by_index = combine_dfs_for_all_stocks_by_index(all_splitted_stocks_dict, 'train_dfs', adjusted_tickers)
    combined_test_dfs_for_all_stocks_by_index = combine_dfs_for_all_stocks_by_index(all_splitted_stocks_dict, 'test_dfs', adjusted_tickers)

    # TODO: normalization should be per stock per df index
    # TODO: scoring should be per stock per df index
    # TODO: when I see in test (and train) a row from 'QQQ' index 77, I'll get the correlations of QQQ index 77 for that row.
    # TODO: scoring will not be holistic for all stocks, but per stock.
    normalized_train_dfs, train_scalers = normalize_dfs(combined_train_dfs_for_all_stocks_by_index, columns_to_normalize)
    # TODO: Save the last scaler for later use. Done
    dump(train_scalers[-1], 'last_train_scaler.gz')

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


    for current_index in range(len(normalized_train_dfs_with_scores)):
        save_create_csv('normalized_train_dfs_with_scores', f'norm_train_df_{current_index}', normalized_train_dfs_with_scores[current_index])
    for current_index in range(len(normalized_test_dfs_with_scores)):
        save_create_csv('normalized_test_dfs_with_scores', f'norm_test_df_{current_index}', normalized_test_dfs_with_scores[current_index])

    # normalized_train_dfs_with_scores = get_dfs_for_all_csvs_in_directory('normalized_train_dfs_with_scores')
    # normalized_test_dfs_with_scores = get_dfs_for_all_csvs_in_directory('normalized_test_dfs_with_scores')

    normalized_train_dfs_with_scores = construct_categorical_cols_for_dfs(normalized_train_dfs_with_scores, adjusted_tickers)
    normalized_test_dfs_with_scores = construct_categorical_cols_for_dfs(normalized_test_dfs_with_scores, adjusted_tickers)


    feature_col_names = get_feature_col_names()
    label_col_name = 'action_return_on_signal_index_categorical'
    all_cols = feature_col_names + [label_col_name]

    clean_train_dfs = drop_rows_with_na_values_from_dfs(normalized_train_dfs_with_scores, all_cols)
    clean_test_dfs = drop_rows_with_na_values_from_dfs(normalized_test_dfs_with_scores, all_cols)

    clean_test_dfs = select_best_actions_using_machine_learning(clean_train_dfs, clean_test_dfs, feature_col_names, label_col_name)

    for current_index in range(len(clean_test_dfs)):
        save_create_csv('test_dfs_with_scores', f'test_df_{current_index}',
                        get_df_without_norm_columns(clean_test_dfs[current_index]))
    all_test_df_with_scores = pd.concat(clean_test_dfs)
    save_create_csv('test_dfs_with_scores', 'all_test_df_with_scores', all_test_df_with_scores)
