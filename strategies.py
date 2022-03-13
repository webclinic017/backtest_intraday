#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from signals import check_volume_high_enough, check_not_earnings_days, check_atr_volatility_low_enough, \
    check_additional_positive_indicators, check_bullish_mas_slopes, check_bearish_mas_slopes
from datetime import datetime


def get_position_direction_and_index(df, i, signal_column_name, in_position_column):
    for j in range(i, 0, -1):
        if df[in_position_column][j] != True:
            return df[signal_column_name][j + 1], j + 1
    return 'ERR', -1


def get_position_direction_and_index_using_signal(df, i, signal_column_name):
    for j in range(i-1, 0, -1):
        if df[signal_column_name][j] is None:
            continue
        elif df[signal_column_name][j] is not None:
            return df[signal_column_name][j], j
    return 'ERR', -1


def exit_bullish(stock_df, current_index, signal_index, trigger_column, time_based_exit=False):
    df = stock_df.copy()
    df.at[current_index, 'exits'] = 'Exit Buy'
    entry_price = df.at[current_index, 'entry_price']
    df.at[current_index, 'action_return'] = (df[trigger_column][current_index] - entry_price) / entry_price
    df.at[signal_index, 'action_return_on_signal_index'] = df['action_return'][current_index]
    df.at[signal_index, 'price_change_for_corr'] = df['action_return'][current_index]
    df.at[signal_index, 'action_length'] = current_index - signal_index
    df.at[current_index, 'in_position'] = False
    if time_based_exit:
        df.at[current_index, 'time_based_exit'] = True
    return df


def exit_bullish_avg(stock_df, current_index, signal_index, time_based_exit=False):
    df = stock_df.copy()
    df.at[current_index, 'exits'] = 'Exit Buy'
    entry_price = df.at[current_index, 'entry_price']
    exit_price = (df.at[current_index, 'High'] + df.at[current_index, 'Low']) / 2
    df.at[current_index, 'action_return'] = (exit_price - entry_price) / entry_price
    df.at[signal_index, 'action_return_on_signal_index'] = df['action_return'][current_index]
    df.at[signal_index, 'price_change_for_corr'] = df['action_return'][current_index]
    df.at[signal_index, 'action_length'] = current_index - signal_index
    df.at[current_index, 'in_position'] = False
    if time_based_exit:
        df.at[current_index, 'time_based_exit'] = True
    return df


def exit_bearish(stock_df, current_index, signal_index, trigger_column, time_based_exit=False):
    df = stock_df.copy()
    df.at[current_index, 'exits'] = 'Exit Sell'
    entry_price = df.at[current_index, 'entry_price']
    df.at[current_index, 'action_return'] = -(df[trigger_column][current_index] - entry_price) / entry_price
    df.at[signal_index, 'action_return_on_signal_index'] = df['action_return'][current_index]
    df.at[signal_index, 'price_change_for_corr'] = -df['action_return'][current_index]
    df.at[signal_index, 'action_length'] = current_index - signal_index
    df.at[current_index, 'in_position'] = False
    if time_based_exit:
        df.at[current_index, 'time_based_exit'] = True
    return df


def exit_bearish_avg(stock_df, current_index, signal_index, time_based_exit=False):
    df = stock_df.copy()
    df.at[current_index, 'exits'] = 'Exit Sell'
    exit_price = (df.at[current_index, 'High'] + df.at[current_index, 'Low']) / 2
    entry_price = df.at[current_index, 'entry_price']
    df.at[current_index, 'action_return'] = -(exit_price - entry_price) / entry_price
    df.at[signal_index, 'action_return_on_signal_index'] = df['action_return'][current_index]
    df.at[signal_index, 'price_change_for_corr'] = -df['action_return'][current_index]
    df.at[signal_index, 'action_length'] = current_index - signal_index
    df.at[current_index, 'in_position'] = False
    if time_based_exit:
        df.at[current_index, 'time_based_exit'] = True
    return df


def check_early_in_trend(df, signal_column_name, i, signal_value, period_check):
    if i < period_check:
        return False
    for j in range(i, i - period_check, -1):
        if df[signal_column_name][j] != signal_value:
            return True
    return False


def calculate_exits_column_by_atr_and_prev_max_min(stock_df, prev_max_min_periods, ticker, time_period='intraday'):
    # time_period = 'intraday' / 'daily'
    df = stock_df.copy()
    df['exits'] = None
    df['action_return'] = None
    df['position_id'] = None
    df['action_return_on_signal_index'] = None
    df['current_stop_loss'] = None
    df['current_profit_taker'] = None
    df['entry_price'] = None
    df['in_position'] = None
    df['signal'] = None
    df['action_length'] = None
    df['profit_potential'] = None
    df['loss_potential'] = None
    df['time_based_exit'] = None
    position_counter = 1
    for i in range(len(df)):
        if i > 1:
            current_date = df.at[i, 'Date']
            # TODO: Had these lines converting a string to date, but currently having a timestamp there so maybe i dont need these lines anyway
            # try:
            #     current_date = datetime.strptime(df.at[i, 'Date'], '%Y-%m-%d %H:%M:%S')
            # except Exception as e:
            #     print(f'could not apply algorithm for {ticker} because of index {i} with error {e}')
            #     continue
            # check in position
            if df.at[i - 1, 'in_position']:
                df.at[i, 'current_stop_loss'] = df.at[i - 1, 'current_stop_loss']
                df.at[i, 'current_profit_taker'] = df.at[i - 1, 'current_profit_taker']
                df.at[i, 'entry_price'] = df.at[i - 1, 'entry_price']
                df.at[i, 'in_position'] = df.at[i - 1, 'in_position']
                df.at[i, 'position_id'] = df.at[i - 1, 'position_id']
                # check for exit
                signal_direction, signal_index = get_position_direction_and_index(df, i, 'signal_direction', 'in_position')
                if signal_direction == 'positive' and df.at[i, 'current_profit_taker'] <= df.at[i, 'Open']:
                    df = exit_bullish(df, i, signal_index, 'Open')  # exit open
                    continue
                if signal_direction == 'positive' and df.at[i, 'current_profit_taker'] <= df.at[i, 'High']:
                    df = exit_bullish(df, i, signal_index, 'current_profit_taker')  # exit pt
                    continue
                if signal_direction == 'positive' and df.at[i, 'current_stop_loss'] >= df.at[i, 'Open']:
                    df = exit_bullish(df, i, signal_index, 'Open')  # exit open
                    continue
                if signal_direction == 'positive' and df.at[i, 'current_stop_loss'] >= df.at[i, 'Low']:
                    df = exit_bullish(df, i, signal_index, 'current_stop_loss')  # exit sl
                    continue
                # # TODO: Trying with a constant stop loss of 2.5% - delete if not relevant
                # if signal_direction == 'positive' and df.at[i, 'entry_price']*0.975 >= df.at[i, 'Open']:
                #     df = exit_bullish(df, i, signal_index, 'Open')  # exit open
                #     continue
                # # TODO: Trying with a constant stop loss of 2.5% - delete if not relevant
                # if signal_direction == 'positive' and df.at[i, 'entry_price']*0.975 >= df.at[i, 'Low']:
                #     df = exit_bullish(df, i, signal_index, 'current_stop_loss')  # exit sl
                #     continue
                # TODO: TIME BASED EXIT - delete if not relevant
                if (time_period == 'intraday' and current_date.hour >= 16) or (time_period == 'daily' and (i - signal_index) >= 5):
                    if signal_direction == 'positive':
                        df = exit_bullish(df, i, signal_index, 'Close', True)
                        continue
                    if signal_direction == 'negative':
                        df = exit_bearish(df, i, signal_index, 'Close', True)
                        continue
                # TODO: TIME BASED EXIT - delete if not relevant. check loss or high profit mid-action length
                # if signal_direction == 'positive' and (i - signal_index) >= 3 and (df.at[i, 'Close'] < df.at[i, 'entry_price'] or (df.at[i, 'Close'] - df.at[i, 'entry_price']) / df.at[i, 'entry_price'] > 0.025):
                #     df = exit_bullish(df, i, signal_index, 'Close', True)  # exit at end of day
                #     continue
                if signal_direction == 'negative' and df.at[i, 'current_stop_loss'] <= df.at[i, 'Open']:
                    df = exit_bearish(df, i, signal_index, 'Open')  # exit open
                    continue
                if signal_direction == 'negative' and df.at[i, 'current_stop_loss'] <= df.at[i, 'High']:
                    df = exit_bearish(df, i, signal_index, 'current_stop_loss')  # exit sl
                    continue
                if signal_direction == 'negative' and df.at[i, 'current_profit_taker'] >= df.at[i, 'Open']:
                    df = exit_bearish(df, i, signal_index, 'Open')  # exit open
                    continue
                if signal_direction == 'negative' and df.at[i, 'current_profit_taker'] >= df.at[i, 'Low']:
                    df = exit_bearish(df, i, signal_index, 'current_profit_taker')  # exit pt
                    continue
                # check for moving stop loss
                if signal_direction == 'positive':
                    if (df.at[i, 'current_profit_taker'] - df.at[i, 'entry_price']) * 0.75 <= df.at[i, 'Close'] - \
                            df.at[i, 'entry_price']:
                        # new stop_loss is max between close-0.5atr and close+reward/2
                        df.at[i, 'current_stop_loss'] = max(df.at[i, 'Close'] - 0.5 * df.at[i, 'atr'],
                                                         (df.at[i, 'current_profit_taker'] + df.at[i, 'entry_price']) / 2)
                        df.at[i, 'current_profit_taker'] = df.at[i, 'current_profit_taker'] + df.at[i, 'atr']
                elif signal_direction == 'negative':
                    if (df.at[i, 'current_profit_taker'] - df.at[i, 'entry_price']) * 0.75 >= df.at[i, 'Close'] - \
                            df.at[i, 'entry_price']:
                        # new stop_loss is min between close+0.5atr and close+reward/2
                        df.at[i, 'current_stop_loss'] = min(df.at[i, 'Close'] + 0.5 * df.at[i, 'atr'],
                                                         (df.at[i, 'current_profit_taker'] + df.at[i, 'entry_price']) / 2)
                        df.at[i, 'current_profit_taker'] = df.at[i, 'current_profit_taker'] - df.at[i, 'atr']
            # if not in position
            elif not df.at[i - 1, 'in_position']:
                # TODO: enter only high volume positions. delete if irrelevant
                # if df.at[i, '50_ma_volume'] < 13000: # enter only positions where 50_ma_volume is higher than 13,000$
                #     continue
                # check if i should enter a bullish position
                if df.at[i, 'signal_direction'] == 'positive':
                    df.at[i, 'entry_price'] = df.at[i, 'Close']
                    df.at[i, 'current_profit_taker'] = max(df['High'].rolling(prev_max_min_periods, 0).max()[i], df.at[i, 'entry_price'] + df.at[i, 'atr'] * 2)
                    df.at[i, 'current_stop_loss'] = min(df['Low'].rolling(5, 0).min()[i], df.at[i, 'entry_price'] - df.at[i, 'atr'])
                    df.at[i, 'profit_potential'] = (df.at[i, 'current_profit_taker'] - df.at[i, 'entry_price']) / df.at[i, 'entry_price']
                    df.at[i, 'loss_potential'] = (df.at[i, 'current_stop_loss'] - df.at[i, 'entry_price']) / df.at[i, 'entry_price']
                    if df.at[i, 'current_profit_taker'] - df.at[i, 'entry_price'] >= 2 * (
                            df.at[i, 'entry_price'] - df.at[i, 'current_stop_loss']):
                        if (time_period == 'intraday' and (current_date.hour < 12) and (current_date.hour == 9 and current_date.minute >= 30 or current_date.hour > 9)) or time_period == 'daily':
                            # enter position
                            df.at[i, 'in_position'] = True
                            df.at[i, 'signal'] = 'Bullish'
                            df.at[i, 'position_id'] = f'{position_counter}_{ticker}'
                            position_counter += 1
                        else:
                            df.at[i, 'in_position'] = False
                    else:
                        df.at[i, 'in_position'] = False
                    continue
                # check if i should enter a bearish position
                if df.at[i, 'signal_direction'] == 'negative':
                    df.at[i, 'entry_price'] = df.at[i, 'Close']
                    df.at[i, 'current_profit_taker'] = df['Low'].rolling(int(prev_max_min_periods / 2), 0).min()[i]
                    df.at[i, 'current_stop_loss'] = df.at[i, 'entry_price'] + df.at[i, 'atr']
                    df.at[i, 'profit_potential'] = abs(df.at[i, 'current_profit_taker'] - df.at[i, 'entry_price']) / df.at[i, 'entry_price']
                    df.at[i, 'loss_potential'] = -(df.at[i, 'current_stop_loss'] - df.at[i, 'entry_price']) / df.at[i, 'entry_price']
                    if abs(df.at[i, 'current_profit_taker'] - df.at[i, 'entry_price']) >= 2 * abs(
                            df.at[i, 'entry_price'] - df.at[i, 'current_stop_loss']):
                        if (time_period == 'intraday' and (current_date.hour < 12) and (
                                current_date.hour == 9 and current_date.minute >= 30 or current_date.hour > 9)) or time_period == 'daily':
                            # enter position
                            df.at[i, 'in_position'] = True
                            df.at[i, 'signal'] = 'Bearish'
                            df.at[i, 'position_id'] = f'{position_counter}_{ticker}'
                            position_counter += 1
                        else:
                            df.at[i, 'in_position'] = False
                    else:
                        df.at[i, 'in_position'] = False
                    continue
    return df


def calculate_returns_for_df_based_on_signals_alone(stock_df, ticker, time_period='intraday'):
    # time_period = 'intraday' / 'daily'
    df = stock_df.copy()
    df['exits'] = None
    df['action_return'] = None
    df['position_id'] = None
    df['action_return_on_signal_index'] = None
    df['entry_price'] = None
    df['in_position'] = None
    df['signal'] = None
    df['action_length'] = None
    df['time_based_exit'] = None
    df['ticker'] = ticker
    position_counter = 1
    for i in range(len(df)):
        if i > 10:
            current_date = df.at[i, 'Date']
            # check in position
            if df.at[i - 1, 'in_position']:
                df.at[i, 'entry_price'] = df.at[i - 1, 'entry_price']
                df.at[i, 'in_position'] = df.at[i - 1, 'in_position']
                df.at[i, 'position_id'] = df.at[i - 1, 'position_id']
                # check for exit
                position_signal_direction, position_signal_index = get_position_direction_and_index_using_signal(df, i, 'signal')
                curr_signal_direction = df.at[i, 'signal_direction']
                if position_signal_direction == 'Bullish' and (curr_signal_direction == 'negative' or (check_bullish_mas_slopes(df, '5_ma', '8_ma', '13_ma', i-2, i-10) and not check_bullish_mas_slopes(df, '5_ma_slope', '8_ma_slope', '13_ma_slope', i, i-1))):
                    df = exit_bullish_avg(df, i, position_signal_index)
                    # TODO: Removed continue in order to detect same day exit enter
                    # continue
                if position_signal_direction == 'Bearish' and (curr_signal_direction == 'positive' or not check_bearish_mas_slopes(df, '5_ma_slope', '8_ma_slope', '13_ma_slope', i, i-2)):
                    df = exit_bearish_avg(df, i, position_signal_index)
                    # TODO: Removed continue in order to detect same day exit enter
                    # continue
                # TODO: TIME BASED EXIT - delete if not relevant
                if time_period == 'intraday' and current_date.hour >= 15 and current_date.minute >= 40:
                    if position_signal_direction == 'Bullish':
                        df = exit_bullish_avg(df, i, position_signal_index, True)
                        continue
                    elif position_signal_direction == 'Bearish':
                        df = exit_bearish_avg(df, i, position_signal_index, True)
                        continue
            # was elif not in position
            # if not df.at[i - 1, 'in_position']:
            if not df.at[i, 'in_position']:
                # TODO: enter only high volume positions. delete if irrelevant
                # if df.at[i, '50_ma_volume'] < 13000: # enter only positions where 50_ma_volume is higher than 13,000$
                #     continue
                # check if i should enter a bullish position
                if df.at[i, 'signal_direction'] == 'positive':
                    if (time_period == 'intraday' and (current_date.hour == 15 and current_date.minute <= 20 or current_date.hour < 15) and (current_date.hour == 10 and current_date.minute >= 15 or current_date.hour > 10)) or time_period == 'daily':
                        # enter position
                        df.at[i, 'entry_price'] = df.at[i, 'Close']
                        df.at[i, 'in_position'] = True
                        df.at[i, 'signal'] = 'Bullish'
                        df.at[i, 'position_id'] = f'{position_counter}_{ticker}'
                        position_counter += 1
                    else:
                        df.at[i, 'in_position'] = False
                    continue
                # check if i should enter a bearish position
                if df.at[i, 'signal_direction'] == 'negative':
                    if (time_period == 'intraday' and (current_date.hour == 15 and current_date.minute <= 20 or current_date.hour < 15) and (current_date.hour == 10 and current_date.minute >= 15 or current_date.hour > 10)) or time_period == 'daily':
                        # enter position
                        df.at[i, 'entry_price'] = df.at[i, 'Close']
                        df.at[i, 'in_position'] = True
                        df.at[i, 'signal'] = 'Bearish'
                        df.at[i, 'position_id'] = f'{position_counter}_{ticker}'
                        position_counter += 1
                    else:
                        df.at[i, 'in_position'] = False
                    continue
    return df
