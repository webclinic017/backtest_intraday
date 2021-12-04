#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from signals import check_volume_high_enough, check_not_earnings_days, check_atr_volatility_low_enough, \
    check_additional_positive_indicators


def get_position_direction_and_index(df, i, signal_column_name, in_position_column):
    for j in range(i, 0, -1):
        if df[in_position_column][j] != True:
            return df[signal_column_name][j + 1], j + 1
    return 'ERR', -1


def exit_bullish(stock_df, current_index, signal_index, trigger_column, time_based_exit=False):
    df = stock_df.copy()
    df.at[current_index, 'exits'] = 'Exit Buy'
    df.at[current_index, 'action_return'] = (df[trigger_column][current_index] - df['Close'][signal_index]) / df['Close'][
        signal_index]
    df.at[signal_index, 'action_return_on_signal_index'] = df['action_return'][current_index]
    df.at[signal_index, 'action_length'] = current_index - signal_index
    df.at[current_index, 'in_position'] = False
    if time_based_exit:
        df.at[current_index, 'time_based_exit'] = True
    return df


def exit_bearish(stock_df, current_index, signal_index, trigger_column, time_based_exit=False):
    df = stock_df.copy()
    df.at[current_index, 'exits'] = 'Exit Sell'
    df[current_index, 'action_return'] = -(df[trigger_column][current_index] - df['Close'][signal_index]) / df['Close'][
        signal_index]
    df.at[signal_index, 'action_return_on_signal_index'] = df['action_return'][current_index]
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


def calculate_exits_column_by_atr_and_prev_max_min(stock_df, prev_max_min_periods, ticker):
    df = stock_df.copy()
    df['exits'] = ''
    df['action_return'] = ''
    df['position_id'] = ''
    df['action_return_on_signal_index'] = ''
    df['current_stop_loss'] = ''
    df['current_profit_taker'] = ''
    df['entry_price'] = ''
    df['in_position'] = ''
    df['signal'] = ''
    df['action_length'] = ''
    df['profit_potential'] = ''
    df['loss_potential'] = ''
    df['time_based_exit'] = ''
    position_counter = 1
    for i in range(len(df)):
        if i > 1:
            # check in position
            if df.at[i - 1, 'in_position']:
                df.at[i, 'current_stop_loss'] = df.at[i - 1, 'current_stop_loss']
                df.at[i, 'current_profit_taker'] = df.at[i - 1, 'current_profit_taker']
                df.at[i, 'entry_price'] = df.at[i - 1, 'entry_price']
                df.at[i, 'in_position'] = df.at[i - 1, 'in_position']
                df.at[i, 'position_id'] = df.at[i - 1, 'position_id']
                # check for exit
                signal_direction, signal_index = get_position_direction_and_index(df, i, 'signal_direction', 'in_position')
                # if signal_direction == 'positive' and df.at[i, 'current_profit_taker'] <= df.at[i, 'Open']:
                #     df = exit_bullish(df, i, signal_index, 'Open')  # exit open
                #     continue
                # if signal_direction == 'positive' and df.at[i, 'current_profit_taker'] <= df.at[i, 'High']:
                #     df = exit_bullish(df, i, signal_index, 'current_profit_taker')  # exit pt
                #     continue
                # if signal_direction == 'positive' and df.at[i, 'current_stop_loss'] >= df.at[i, 'Open']:
                #     df = exit_bullish(df, i, signal_index, 'Open')  # exit open
                #     continue
                # if signal_direction == 'positive' and df.at[i, 'current_stop_loss'] >= df.at[i, 'Low']:
                #     df = exit_bullish(df, i, signal_index, 'current_stop_loss')  # exit sl
                #     continue
                # # TODO: Trying with a constant stop loss of 2.5% - delete if not relevant
                # if signal_direction == 'positive' and df.at[i, 'entry_price']*0.975 >= df.at[i, 'Open']:
                #     df = exit_bullish(df, i, signal_index, 'Open')  # exit open
                #     continue
                # # TODO: Trying with a constant stop loss of 2.5% - delete if not relevant
                # if signal_direction == 'positive' and df.at[i, 'entry_price']*0.975 >= df.at[i, 'Low']:
                #     df = exit_bullish(df, i, signal_index, 'current_stop_loss')  # exit sl
                #     continue
                # TODO: TIME BASED EXIT - delete if not relevant
                if signal_direction == 'positive' and (i - signal_index) >= 5:
                    df = exit_bullish(df, i, signal_index, 'Close', True)  # exit at end of day
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
                # check if i should enter a bullish position
                if df.at[i, 'signal_direction'] == 'positive':
                    df.at[i, 'entry_price'] = df.at[i, 'Close']
                    df.at[i, 'current_profit_taker'] = max(df['High'].rolling(prev_max_min_periods).max()[i], df.at[i, 'entry_price'] + df.at[i, 'atr'] * 2)
                    df.at[i, 'current_stop_loss'] = min(df['Low'].rolling(5).min()[i], df.at[i, 'entry_price'] - df.at[i, 'atr'])
                    df.at[i, 'profit_potential'] = (df.at[i, 'current_profit_taker'] - df.at[i, 'entry_price']) / df.at[i, 'entry_price']
                    df.at[i, 'loss_potential'] = (df.at[i, 'current_stop_loss'] - df.at[i, 'entry_price']) / df.at[i, 'entry_price']
                    if df.at[i, 'current_profit_taker'] - df.at[i, 'entry_price'] >= 2 * (
                            df.at[i, 'entry_price'] - df.at[i, 'current_stop_loss']):
                        # enter position
                        df.at[i, 'in_position'] = True
                        df.at[i, 'signal'] = 'Bullish'
                        df.at[i, 'position_id'] = f'{position_counter}_{ticker}'
                        position_counter += 1
                    else:
                        df.at[i, 'in_position'] = False
                    continue
                # check if i should enter a bearish position
                if df.at[i, 'signal_direction'] == 'negative' and check_not_earnings_days(df, i):
                    df.at[i, 'entry_price'] = df['Close'][i]
                    df.at[i, 'current_profit_taker'] = df['Low'].rolling(int(prev_max_min_periods / 2)).min()[i]
                    df.at[i, 'current_stop_loss'] = df.at[i, 'entry_price'] + df.at[i, 'atr']
                    df.at[i, 'profit_potential'] = abs(df.at[i, 'current_profit_taker'] - df.at[i, 'entry_price']) / df.at[i, 'entry_price']
                    df.at[i, 'loss_potential'] = -(df.at[i, 'current_stop_loss'] - df.at[i, 'entry_price']) / df.at[i, 'entry_price']
                    if abs(df.at[i, 'current_profit_taker'] - df.at[i, 'entry_price']) >= 2 * abs(
                            df.at[i, 'entry_price'] - df.at[i, 'current_stop_loss']):
                        # enter position
                        df.at[i, 'in_position'] = True
                        df.at[i, 'signal'] = 'Bearish'
                        df.at[i, 'position_id'] = f'{position_counter}_{ticker}'
                        position_counter += 1
                    else:
                        df.at[i, 'in_position'] = False
                    continue
    return df
