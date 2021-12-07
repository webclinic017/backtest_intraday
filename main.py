import pandas as pd
import numpy as np
import seaborn as seaborn

from data_fetcher import get_sp500_list, get_data_dict_for_all_stocks_in_directory, get_data_dict_for_multiple_stocks, \
    get_data_for_stock, get_stock_data_trade_daily_alpha_vantage
from strategies import calculate_exits_column_by_atr_and_prev_max_min
from indicators import get_ma_column_for_stock, get_distance_between_columns_for_stock, \
    get_adx_column_for_stock, rsi, stochastic, get_ATR_column_for_stock, get_volatility_from_atr, \
    get_macd_columns_for_stock, normalize_columns, get_beta_column, get_breakout_column_for_stock
from signals import indicators_mid_levels_signal, parabolic_trending_n_periods, cross_20_ma, cross_50_ma, joint_signal, \
    macd_cross_0_signal, macd_signal_cross_signal, joint_macd_signal_cross_signal, joint_macd_cross_0_signal, \
    awesome_oscilator, calculate_correl_score_series_for_df, cumulative_rsi_signal
import time
import matplotlib.pyplot as plt
import seaborn as sns
import datetime



# See PyCharm help at https://www.jetbrains.com/help/pycharm/

tickers = get_sp500_list()

adjusted_tickers = [elem for elem in tickers if elem != 'GOOG' and elem != 'DUK' and elem != 'HLT' and elem != 'DD' and elem != 'CMCSA' and elem != 'COG' and elem != 'WBA' and elem != 'KMX' and elem != 'ADP' and elem != 'STZ' and elem != 'IQV'] # there were stock splits
adjusted_tickers = [elem for elem in adjusted_tickers if '.' not in elem]
# yahoo finance screener - mega caps only, tech, energey and finance
adjusted_tickers = ['FB', 'AAPL', 'NFLX', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'BAC', 'C', 'TWTR', 'MA', 'TSM', 'V', 'NVDA', 'XOM', 'CVX']
# adjusted_tickers = ['MSFT']

adjusted_tickers = adjusted_tickers + ['SPY', 'QQQ', 'IWM']
# adjusted_tickers = adjusted_tickers + ['SPY']

stocks_dict = get_data_dict_for_multiple_stocks(adjusted_tickers, time)
# spy_df = stocks_dict['SPY']

# stocks_dict, adjusted_tickers = get_data_dict_for_all_stocks_in_directory('stocks_csvs_new')

# adjusted_tickers = ['FB', 'AAPL', 'NFLX', 'GOOGL', 'MSFT', 'AMZN', 'SPY', 'QQQ', 'IWM']

# stocks_dict = { tick: stocks_dict[tick].iloc[-(252*4):].reset_index(drop=True) for tick in adjusted_tickers }

for ticker in adjusted_tickers:
    if ticker not in stocks_dict:
        adjusted_tickers.remove(ticker)

all_stocks_data_df = pd.DataFrame()
all_stocks_data_df['ticker'] = adjusted_tickers

start = time.time()

for ticker in adjusted_tickers:
    stocks_dict[ticker]['10_ma'] = get_ma_column_for_stock(stocks_dict[ticker], 'Close', 10)
    stocks_dict[ticker]['20_ma'] = get_ma_column_for_stock(stocks_dict[ticker], 'Close', 20)
    stocks_dict[ticker]['50_ma'] = get_ma_column_for_stock(stocks_dict[ticker], 'Close', 50)
    stocks_dict[ticker]['200_ma'] = get_ma_column_for_stock(stocks_dict[ticker], 'Close', 200)
    stocks_dict[ticker]['10_ma_volume'] = get_ma_column_for_stock(stocks_dict[ticker], 'Volume', 10)
    stocks_dict[ticker]['20_ma_volume'] = get_ma_column_for_stock(stocks_dict[ticker], 'Volume', 20)
    stocks_dict[ticker]['50_ma_volume'] = get_ma_column_for_stock(stocks_dict[ticker], 'Volume', 50)
    stocks_dict[ticker]['10_ma_volume_break'] = get_breakout_column_for_stock(stocks_dict[ticker], 'Volume', '10_ma_volume', '10_ma_volume_break')
    stocks_dict[ticker]['20_ma_volume_break'] = get_breakout_column_for_stock(stocks_dict[ticker], 'Volume', '20_ma_volume', '20_ma_volume_break')
    stocks_dict[ticker]['50_ma_volume_break'] = get_breakout_column_for_stock(stocks_dict[ticker], 'Volume', '50_ma_volume', '50_ma_volume_break')
    stocks_dict[ticker]['10_beta_SPY'] = get_beta_column(stocks_dict[ticker], stocks_dict['SPY'], 10)
    stocks_dict[ticker]['50_beta_SPY'] = get_beta_column(stocks_dict[ticker], stocks_dict['SPY'], 50)
    # stocks_dict[ticker]['10_beta_QQQ'] = get_beta_column(stocks_dict[ticker], stocks_dict['QQQ'], 10)
    # stocks_dict[ticker]['50_beta_QQQ'] = get_beta_column(stocks_dict[ticker], stocks_dict['QQQ'], 50)
    # stocks_dict[ticker]['10_beta_IWM'] = get_beta_column(stocks_dict[ticker], stocks_dict['IWM'], 10)
    # stocks_dict[ticker]['50_beta_IWM'] = get_beta_column(stocks_dict[ticker], stocks_dict['IWM'], 50)
    stocks_dict[ticker]['median'] = (stocks_dict[ticker]['High'] + stocks_dict[ticker]['Low']) / 2
    stocks_dict[ticker]['ma_med_5'] = get_ma_column_for_stock(stocks_dict[ticker], 'median', 5)
    stocks_dict[ticker]['ma_med_34'] = get_ma_column_for_stock(stocks_dict[ticker], 'median', 34)
    stocks_dict[ticker]['awesome_osc'] = stocks_dict[ticker]['ma_med_5'] - stocks_dict[ticker]['ma_med_34']
    stocks_dict[ticker]['median_ratio'] = stocks_dict[ticker]['median'] / stocks_dict[ticker]['Close']
    stocks_dict[ticker]['ma_med_5_ratio'] = stocks_dict[ticker]['ma_med_5'] / stocks_dict[ticker]['Close']
    stocks_dict[ticker]['ma_med_34_ratio'] = stocks_dict[ticker]['ma_med_34'] / stocks_dict[ticker]['Close']
    stocks_dict[ticker]['macd'], stocks_dict[ticker]['macd_signal'] = get_macd_columns_for_stock(stocks_dict[ticker], 12, 26, 9)
    stocks_dict[ticker]['atr'] = get_ATR_column_for_stock(stocks_dict[ticker], 14)
    stocks_dict[ticker]['distance_from_10_ma'] = get_distance_between_columns_for_stock(stocks_dict[ticker], 'Close', '10_ma')
    stocks_dict[ticker]['adx'], stocks_dict[ticker]['+di'], stocks_dict[ticker]['-di'] = get_adx_column_for_stock(stocks_dict[ticker], 14)
    stocks_dict[ticker]['adx_ma_med_5_rat'] = stocks_dict[ticker]['adx']*stocks_dict[ticker]['ma_med_5_ratio']
    stocks_dict[ticker]['rsi'] = rsi(stocks_dict[ticker], 2) # changed from 14
    stocks_dict[ticker]['stochastic_k'], stocks_dict[ticker]['stochastic_d'] = stochastic(stocks_dict[ticker], 14, 3)
    stocks_dict[ticker]['atr_volatility'], stocks_dict[ticker]['atr_volatility_ma'] = get_volatility_from_atr(stocks_dict[ticker], 14)
    stocks_dict[ticker]['signal_type'] = ''
    stocks_dict[ticker]['signal_direction'] = ''
    stocks_dict[ticker]['indicators_mid_levels_signal'] = ''
    stocks_dict[ticker]['indicators_mid_level_direction'] = ''
    stocks_dict[ticker]['cross_20_signal'] = ''
    stocks_dict[ticker]['cross_20_direction'] = ''
    stocks_dict[ticker]['cross_50_signal'] = ''
    stocks_dict[ticker]['cross_50_direction'] = ''
    # signal_type and signal_direction columns are the columns that determine the actual orders!
    # stocks_dict[ticker] = indicators_mid_levels_signal(stocks_dict[ticker], 'indicators_mid_level_direction', 'indicators_mid_levels_signal')
    # stocks_dict[ticker] = cross_20_ma(stocks_dict[ticker], 'cross_20_direction', 'cross_20_signal')
    # stocks_dict[ticker] = cross_50_ma(stocks_dict[ticker], 'cross_50_direction', 'cross_50_signal')

    # stocks_dict[ticker] = joint_signal(stocks_dict[ticker], 'signal_direction', 'signal_type')
    stocks_dict[ticker] = awesome_oscilator(stocks_dict[ticker], 'signal_direction', 'signal_type')
    # stocks_dict[ticker] = cumulative_rsi_signal(stocks_dict[ticker], 'signal_direction', 'signal_type')

    stocks_dict[ticker] = calculate_exits_column_by_atr_and_prev_max_min(stocks_dict[ticker], 70, ticker)
    stocks_dict[ticker] = stocks_dict[ticker].reset_index()
    stocks_dict[ticker].to_csv(f'stocks_csvs_new/{ticker}_engineered.csv', index=False)
    # stocks_dict[ticker].tail(1000).plot(x="Date", y=["Close", "50_ma"])
    # plt.show()

end = time.time()
print(f'time for processing all stocks: {end - start}')

# add data to some whole stocks data df
all_stocks_data_df['average_action_p_l'] = ''
all_stocks_data_df['median_action_p_l'] = ''
all_stocks_data_df['min_action_p_l'] = ''
all_stocks_data_df['max_action_p_l'] = ''
all_stocks_data_df['total_p_l'] = ''
all_stocks_data_df['total_correct_actions'] = ''
all_stocks_data_df['total_wrong_actions'] = ''
all_stocks_data_df['total_actions'] = ''
all_stocks_data_df['total_periods'] = ''
all_stocks_data_df['pct_actions'] = ''
all_stocks_data_df['pct_correct_actions'] = ''
for index, ticker in enumerate(adjusted_tickers):
    print(f'all stocks data: {ticker}')
    all_stocks_data_df['average_action_p_l'][index] = stocks_dict[ticker]['action_return'].replace('', np.nan).mean()
    all_stocks_data_df['median_action_p_l'][index] = stocks_dict[ticker]['action_return'].replace('', np.nan).median()
    all_stocks_data_df['min_action_p_l'][index] = stocks_dict[ticker]['action_return'].replace('', np.nan).min()
    all_stocks_data_df['max_action_p_l'][index] = stocks_dict[ticker]['action_return'].replace('', np.nan).max()
    temp_series_cumprod = (1 + stocks_dict[ticker]['action_return'].replace('', np.nan)).cumprod()
    if temp_series_cumprod.dropna().empty:
        all_stocks_data_df['total_p_l'][index] = 0
    else:
        all_stocks_data_df['total_p_l'][index] = temp_series_cumprod.dropna().iloc[-1] - 1
    all_stocks_data_df['total_correct_actions'][index] = stocks_dict[ticker]['action_return'][stocks_dict[ticker]['action_return'].replace('', np.nan) > 0].count()
    all_stocks_data_df['total_wrong_actions'][index] = stocks_dict[ticker]['action_return'][stocks_dict[ticker]['action_return'].replace('', np.nan) < 0].count()
    all_stocks_data_df['total_actions'][index] = all_stocks_data_df['total_correct_actions'][index] + all_stocks_data_df['total_wrong_actions'][index]
    all_stocks_data_df['total_periods'][index] = len(stocks_dict[ticker])
    all_stocks_data_df['pct_actions'][index] = all_stocks_data_df['total_actions'][index] / len(stocks_dict[ticker])
    if all_stocks_data_df['total_actions'][index] == 0:
        continue
    all_stocks_data_df['pct_correct_actions'][index] = all_stocks_data_df['total_correct_actions'][index] / all_stocks_data_df['total_actions'][index]
all_stocks_data_df.to_csv(f'stocks_csvs_new/all_stocks_data.csv', index=False)


# convert stocks data to one df, normalize, and put together as a dict


all_actions_df = pd.DataFrame()
for index, ticker in enumerate(adjusted_tickers):
    print(f'all actions df: {ticker}')
    current_actions_df = stocks_dict[adjusted_tickers[index]].loc[stocks_dict[adjusted_tickers[index]]['in_position'] != ''].copy()
    if len(current_actions_df) == 0:
        continue
    current_actions_df.loc[:, 'ticker'] = ticker
    # current_actions_df = current_actions_df[current_actions_df['action_return_on_signal_index'] != '']
    if index == 0:
        all_actions_df = current_actions_df
    else:
        all_actions_df = pd.concat([all_actions_df, current_actions_df])


INITIAL_POCKET_VALUE = 100
MAX_POCKETS = 1
pocket_list = [{
    'amount': INITIAL_POCKET_VALUE,
    'in_position': False,
    'ticker': ''
}]
num_initial_pockets = 1
num_entered_positions = 0


def lock_pocket(pocket_list, ticker, num_initial_pockets, num_entered_positions):
    for pocket_index in range(len(pocket_list)):
        if pocket_list[pocket_index]['in_position'] == False:
            num_entered_positions += 1
            pocket_list[pocket_index]['in_position'] = True
            pocket_list[pocket_index]['ticker'] = ticker
            return pocket_index, num_initial_pockets, num_entered_positions, True
    # didnt find unlocked pocket - init a new one

    if len(pocket_list) < MAX_POCKETS:
        pocket_list.append({
            'amount': INITIAL_POCKET_VALUE,
            'in_position': True,
            'ticker': ticker
        })
        num_initial_pockets += 1
    return len(pocket_list) - 1, num_initial_pockets, num_entered_positions, False


def merge_unlocked_pockets(pocket_list):
    sum_unlocked_pockets = 0
    num_unlocked_pockets = 0
    for pocket_index in range(len(pocket_list)):
        if pocket_list[pocket_index]['in_position'] == False:
            sum_unlocked_pockets += pocket_list[pocket_index]['amount']
            num_unlocked_pockets += 1
    for pocket_index in range(len(pocket_list)):
        if pocket_list[pocket_index]['in_position'] == False:
            pocket_list[pocket_index]['amount'] = sum_unlocked_pockets / num_unlocked_pockets
    # new_pocket_list = [pocket for pocket in pocket_list if pocket.get('in_position') == True]
    # new_pocket_list.append({
    #     'amount': sum_unlocked_pockets,
    #     'in_position': False,
    #     'ticker': ''
    # })
    return pocket_list


def unlock_pocket(pocket_list, ticker, pct_gains):
    for pocket_index in range(len(pocket_list)):
        if pocket_list[pocket_index]['in_position'] == True and pocket_list[pocket_index]['ticker'] == ticker:
            pocket_list[pocket_index]['in_position'] = False
            pocket_list[pocket_index]['ticker'] = ''
            pocket_list[pocket_index]['amount'] = (1 + int(pct_gains or 0)) * int(pocket_list[pocket_index]['amount'] or 0)
    return merge_unlocked_pockets(pocket_list)
    # return pocket_list


def get_correls_on_norm_columns(df, cols):
    copied_df = df.copy()
    corr_dict = {}
    copied_df = copied_df.replace(r'^\s*$', np.NaN, regex=True)
    for col in cols:
        corr_dict[f'{col}_norm'] = copied_df['action_return_on_signal_index'].fillna(0).astype(float).corr(copied_df[f'{col}_norm'].fillna(0).astype(float))
    print(f'correlations with action return on signal index summary: {corr_dict}')

    # TODO: winning keys should be calculated on an automated monthly basis and pushed to a db, pulled in orders_notifier (taking into account preventing overfitting and less correlated features)
    # TODO: Update position scores in orders notifier!!!
    winning_keys = ['50_beta_SPY_norm', 'median_ratio_norm', 'awesome_osc_norm', 'rsi_norm', 'atr_volatility_norm']
    winning_corr_dict = {winning_key: corr_dict[winning_key] for winning_key in winning_keys}
    return winning_corr_dict


all_actions_df['pocket_amount'] = ''
all_actions_df['pocket_index'] = ''
all_actions_df['total_pockets_value'] = ''
all_actions_df['position_score'] = ''
all_actions_df['highest_score_for_day'] = ''
all_actions_df = all_actions_df.sort_values(by=['Date'])
all_actions_df = all_actions_df.reset_index(drop=True)
columns_to_normalize = ['Volume', '10_ma_volume', '20_ma_volume', '50_ma_volume',
                                                    '10_beta_SPY', '50_beta_SPY', 'median_ratio', 'ma_med_5_ratio',
                                                    'ma_med_34_ratio', 'awesome_osc','macd', 'macd_signal',
                                                    'distance_from_10_ma', 'adx', '+di', '-di', 'rsi', 'stochastic_k',
                                                    'stochastic_d', 'atr_volatility', 'atr_volatility_ma']
all_actions_df = normalize_columns(all_actions_df, columns_to_normalize)

only_entrances_df = all_actions_df.copy()[all_actions_df['action_return_on_signal_index'] != '']
correls_dict = get_correls_on_norm_columns(only_entrances_df, columns_to_normalize)


only_entrances_df = calculate_correl_score_series_for_df(only_entrances_df, correls_dict)
total_actions_before_one_per_day = len(only_entrances_df)

# TODO: comment out this section only to calculate correl again >>
best_position_ids = only_entrances_df.sort_values('position_score', ascending=False).drop_duplicates(['Date']).sort_values(['Date']).reset_index(drop=True)['position_id']
only_entrances_df['highest_score_for_day'] = only_entrances_df['position_id'].isin(best_position_ids)
# all_actions_df = all_actions_df[all_actions_df['position_id'].isin(best_position_ids)]
# all_actions_df = all_actions_df.sort_values(by=['Date'])
# all_actions_df = all_actions_df.reset_index(drop=True)
# TODO: << comment out this section only to calculate correl again
for row in range(len(all_actions_df)):
    if all_actions_df.at[row, 'signal'] != '':
        locked_pocket_index, num_initial_pockets, num_entered_positions, entered = lock_pocket(pocket_list, all_actions_df.at[row, 'ticker'], num_initial_pockets, num_entered_positions)
        if entered:
            all_actions_df.at[row, 'pocket_amount'] = pocket_list[locked_pocket_index]['amount']
            all_actions_df.at[row, 'pocket_index'] = locked_pocket_index
    if all_actions_df.at[row, 'exits'] != '':
        pocket_list = unlock_pocket(pocket_list, all_actions_df.at[row, 'ticker'], all_actions_df.at[row, 'action_return'])
        all_actions_df.at[row, 'total_pockets_value'] = sum(item['amount'] for item in pocket_list)
    print(all_actions_df.at[row, 'Date'])
    print(pocket_list)

all_actions_df = only_entrances_df
# TODO: comment out to not show correl plots >>
# all_actions_df['correct'] = all_actions_df['action_return_on_signal_index'] > 0.006
# classes = all_actions_df['correct']
# features = all_actions_df[['Volume_norm', 'ma_volume_norm', 'median_ratio_norm', 'ma_med_5_ratio_norm', 'ma_med_34_ratio_norm', 'awesome_osc_norm',
#                             'macd_norm', 'macd_signal_norm', 'distance_from_10_ma_norm', 'adx_norm', '+di_norm', '-di_norm', 'rsi_norm', 'stochastic_k_norm',
#                             'stochastic_d_norm', 'atr_volatility_norm', 'atr_volatility_ma_norm']]
# data = pd.concat([classes, features.iloc[:,16:]], axis=1)
# data = pd.melt(data, id_vars="correct",
#  var_name="features",
#  value_name='value')
# plt.figure(figsize=(16,8))
# sns.violinplot(x="features", y="value", hue="correct", data=data, split=True,
#  inner="quart")
# plt.xticks(rotation=90)
# TODO: << comment out to not show correl plots


def add_average_gain_for_same_day_col(df):
    df_copy = df.copy().reset_index(drop=True)
    df_copy['average_same_day_gain'] = df_copy['action_return_on_signal_index']
    df_gains = df_copy[['Date', 'average_same_day_gain']]
    df_gains.loc[:, 'average_same_day_gain'] = pd.to_numeric(df_gains['average_same_day_gain'])
    df_gains = df_gains.groupby(by=["Date"]).mean().reset_index()
    for i in range(len(df_copy)):
        df_copy['average_same_day_gain'][i] = df_gains.loc[df_gains['Date'] == df_copy['Date'][i], 'average_same_day_gain'].values[0]
    return df_copy


def add_gap_n_days_col(df, n_days):
    df[f'is_{n_days}_gap'] = False
    last_entered_position_date = ''
    for i in range(len(df)):
        current_date = df.at[i, 'Date']
        print(i)
        print(type(current_date))
        if type(current_date) == str:
            current_date = datetime.datetime.strptime(current_date, '%Y-%m-%d')
        if last_entered_position_date == '':
            last_entered_position_date = current_date
            df.at[i, f'is_{n_days}_gap'] = True
        if df.at[i, 'Volume'] > df.at[i, '10_ma_volume'] and df.at[i, 'position_score'] > 0 and (current_date >= last_entered_position_date + datetime.timedelta(days=n_days) or current_date == last_entered_position_date):
            df.at[i, f'is_{n_days}_gap'] = True
            last_entered_position_date = current_date
    return df


def add_spy_qqq_close(df, spy_df, qqq_df):
    df['spy_close'] = ''
    df['qqq_close'] = ''

    for i in range(len(df)):
        current_date = df.at[i, 'Date']
        df.at[i, 'spy_close'] = spy_df.loc[spy_df['Date'] == current_date, 'Close'].values[0]
        df.at[i, 'qqq_close'] = qqq_df.loc[qqq_df['Date'] == current_date, 'Close'].values[0]
    return df


all_actions_df = add_average_gain_for_same_day_col(all_actions_df)
# all_actions_df = add_gap_n_days_col(all_actions_df, 5)
# all_actions_df = add_spy_qqq_close(all_actions_df, stocks_dict['SPY'], stocks_dict['QQQ'])

all_actions_df.to_csv(f'stocks_csvs_new/all_actions_df.csv', index=False)

sum_init_pockets = num_initial_pockets * INITIAL_POCKET_VALUE
sum_end_pockets = sum(item["amount"] for item in pocket_list)
total_gains = (sum_end_pockets - sum_init_pockets) / sum_init_pockets
print(f'pocket_list: {pocket_list}')
print(f'sum of all initial pockets {sum_init_pockets}')
print(f'current pockets value {sum_end_pockets}')
print(f'total gains pct: {total_gains}')
print(f'number of potential positions: {total_actions_before_one_per_day}')
print(f'number of potential positions after clean for 1 per day: {len(all_actions_df)}')
print(f'number of entered positions: {num_entered_positions}')


def merge_returns_with_same_date(df):
    df_copy = df.copy()
    df_copy.loc[:, 'action_return_on_signal_index'] = pd.to_numeric(df_copy['action_return_on_signal_index'])
    df_copy.loc[:, 'pocket_amount'] = pd.to_numeric(df_copy['pocket_amount'])
    return df_copy.groupby(by=["Date"]).mean()


sp500 = get_data_for_stock('SPY', 'D', time.time(), time).sort_values(by=['Date'])

all_actions_df_merged = merge_returns_with_same_date(all_actions_df[['action_return_on_signal_index', 'Date', 'pocket_amount']]).sort_values(by=['Date'])

algo_gains_df = pd.concat([sp500.set_index('Date')[['Close']], all_actions_df_merged[['action_return_on_signal_index', 'pocket_amount']]], axis=1)
algo_gains_df['action_return_on_signal_index'] = algo_gains_df['action_return_on_signal_index'].fillna(0)
algo_gains_df['pocket_amount'] = algo_gains_df['pocket_amount'].fillna(0)
algo_gains_df = algo_gains_df.loc[algo_gains_df['pocket_amount'] != 0]

STARTER_AMOUNT = 100.0
algo_gains_df['algo_value'] = STARTER_AMOUNT
algo_gains_df['sp_value'] = STARTER_AMOUNT
algo_gains_df = algo_gains_df.reset_index().sort_values(by=['Date'])
for action_index in range(len(algo_gains_df)):
    if action_index == 0:
        algo_gains_df.at[action_index, 'algo_value'] = STARTER_AMOUNT + STARTER_AMOUNT * algo_gains_df.at[action_index, 'action_return_on_signal_index']
        algo_gains_df.at[action_index, 'sp_value'] = STARTER_AMOUNT
        continue
    algo_gains_df.at[action_index, 'algo_value'] = algo_gains_df.at[action_index - 1, 'algo_value'] + algo_gains_df.at[action_index - 1, 'algo_value'] * algo_gains_df.at[action_index, 'action_return_on_signal_index']
    algo_gains_df.at[action_index, 'sp_value'] = algo_gains_df.at[action_index - 1, 'sp_value'] + algo_gains_df.at[action_index - 1, 'sp_value'] * \
                                                 ((algo_gains_df.at[action_index, 'Close'] - algo_gains_df.at[action_index - 1, 'Close']) / algo_gains_df.at[action_index - 1, 'Close'])

algo_gains_df = algo_gains_df.set_index('Date').sort_values(by=['Date'])
algo_gains_df[['sp_value', 'algo_value']].plot(figsize=(16, 8))
plt.show()

algo_gains_df.reset_index().to_csv(f'stocks_csvs_new/algo_gains_df.csv', index=False)

# latest_actions_df = pd.DataFrame()
# for index, ticker in enumerate(adjusted_tickers):
#     if stocks_dict[ticker]['in_position'].iloc[-1] != True:
#         continue
#     current_actions_df = stocks_dict[ticker]
#     current_actions_df['ticker'] = ticker
#     last_position_enter_index = len(current_actions_df)
#     for i in range(len(current_actions_df), 0, -1):
#         if current_actions_df['in_position'][i] != True:
#             last_position_enter_index = i
#             break
#     current_actions_df = current_actions_df.tail(len(current_actions_df) - last_position_enter_index)
#     if index == 0:
#         latest_actions_df = current_actions_df
#     else:
#         latest_actions_df = pd.concat([latest_actions_df, current_actions_df])
#
# latest_actions_df.to_csv(f'stocks_csvs_new/latest_actions_df.csv', index=False)
# finish = 1
