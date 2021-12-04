#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from indicators import normalize_columns
import pandas as pd


def check_non_adx_indicators_before_n_periods(df, i, num_periods, check_term):
    # check_term could be 'ABOVE' or 'BELOW'
    if i < num_periods:
        return True
    if check_term == 'ABOVE':
        if df['distance_from_10_ma'][i - num_periods] > 0 and df['rsi'][i - num_periods] > 50 and df['+di'][i - num_periods] > 25 and df['-di'][i - num_periods] < 25 and df['stochastic_k'][i - num_periods] > 50 and df['stochastic_d'][i - num_periods] > 50:
            return False
    elif check_term == 'BELOW':
        if df['distance_from_10_ma'][i - num_periods] < 0 and df['rsi'][i - num_periods] < 50 and df['+di'][i - num_periods] < 25 and df['-di'][i - num_periods] > 25 and df['stochastic_k'][i - num_periods] < 50 and df['stochastic_d'][i - num_periods]< 50:
            return False
    return True


def check_more_bull_signals(df, i):
    count = 0
    if df['rsi'][i] >= 90:
        count += 1
    if df['stochastic_k'][i] < 60:
        count += 1
    if df['stochastic_d'][i] < 60:
        count += 1
    return count == 0


def check_more_bear_signals(df, i):
    count = 0
    if df['-di'][i] > 40:
        count += 1
    if df['stochastic_k'][i] < 15:
        count += 1
    if df['stochastic_d'][i] < 15:
        count += 1
    if df['atr_volatility'][i] > 0.075 and df['atr_volatility_ma'][i] > 0.075:
        count += 1
    return count == 0


def check_column_trend(df, column_name, i, diff=0):
    if i > 2:
        if df[column_name][i] + diff < df[column_name][i - 1] + diff < df[column_name][i - 2] + diff:
            return 'DOWN'
        if df[column_name][i] - diff > df[column_name][i - 1] - diff > df[column_name][i - 2] - diff:
            return 'UP'
    return 'NO_TREND'


def check_column_below_indicator_last_n_periods(df, indicator_name, column_name, current_index, n):
    i = current_index - 1
    while i >= current_index - n:
        if df.at[i, column_name] < df.at[i, indicator_name]:
            i -= 1
        else:
            return False
    return True


def cross_20_ma(stock_df, signal_direction_column, signal_type_column):
    df = stock_df.copy()
    for i in range(len(df)):
        if i > 1:
            if df['Close'][i] > df['20_ma'][i] and check_column_below_indicator_last_n_periods(df, '20_ma', 'Close', i, 10) == True:
                df.at[i, signal_direction_column] = 'positive'
                df.at[i, signal_type_column] = 'cross_20'
            # elif (df['20_ma'][i] - df['Close'][i]) / df['20_ma'][i] > 0.01 and (df['20_ma'][i-1] - df['Close'][i-1]) / df['20_ma'][i-1] <= 0.01:
            #     df[signal_direction_column][i] = 'negative'
            #     df[signal_type_column][i] = 'cross_20'
    return df


def cross_50_ma(stock_df, signal_direction_column, signal_type_column):
    df = stock_df.copy()
    for i in range(len(df)):
        if i > 1:
            if df['Close'][i] > df['50_ma'][i] and check_column_below_indicator_last_n_periods(df, '50_ma', 'Close', i, 25) == True:
                df.at[i, signal_direction_column] = 'positive'
                df.at[i, signal_type_column] = 'cross_50'
            # elif (df['50_ma'][i] - df['Close'][i]) / df['50_ma'][i] > 0.01 and (df['50_ma'][i-1] - df['Close'][i-1]) / df['50_ma'][i-1] <= 0.01:
            #     df[signal_direction_column][i] = 'negative'
            #     df[signal_type_column][i] = 'cross_50'
    return df


def check_awesome_osc_twin_peaks_in_negative_zone(df, index, period):
    # assuming we have a cross to positive on the index point
    early_min = 0
    later_min = 0
    lock_later_min = False
    for i in range(index - 1, index - period, -1):
        if df['awesome_osc'][i] > 0:
            return False
        if later_min > df['awesome_osc'][i] and not lock_later_min:
            later_min = df['awesome_osc'][i]
        elif later_min < df['awesome_osc'][i] and not lock_later_min:
            lock_later_min = True
        if early_min > df['awesome_osc'][i] and lock_later_min:
            early_min = df['awesome_osc'][i]
        if early_min < later_min:
            return True
    return False


def check_awesome_osc_twin_peaks_in_positive_zone(df, index, period):
    # assuming we have a cross to negative on the index point
    early_max = 0
    later_max = 0
    lock_later_max = False
    for i in range(index - 1, index - period, -1):
        if df['awesome_osc'][i] < 0:
            return False
        if later_max < df['awesome_osc'][i] and not lock_later_max:
            later_max = df['awesome_osc'][i]
        elif later_max > df['awesome_osc'][i] and not lock_later_max:
            lock_later_max = True
        if early_max < df['awesome_osc'][i] and lock_later_max:
            early_max = df['awesome_osc'][i]
        if early_max > later_max:
            return True
    return False


def awesome_oscilator(stock_df, signal_direction_column, signal_type_column):
    df = stock_df.copy()
    for i in range(len(df)):
        if i > 100:
            if df['awesome_osc'][i] > 0 and (df.loc[i-7 : i-1, 'awesome_osc'] < 0).all() and check_awesome_osc_twin_peaks_in_negative_zone(df, i, 80):
                df.at[i, signal_direction_column] = 'positive'
                df.at[i, signal_type_column] = 'awesome_osc'
            # elif df['awesome_osc'][i] < 0 and df['awesome_osc'][i-1] >= 0 and check_awesome_osc_twin_peaks_in_positive_zone(df, i, 80):
            #     df[signal_direction_column][i] = 'negative'
            #     df[signal_type_column][i] = 'awesome_osc'
    return df


def cumulative_rsi_signal(stock_df, signal_direction_column, signal_type_column):
    df = stock_df.copy()
    for i in range(len(df)):
        if i > 200:
            if df.at[i, 'Close'] > df.at[i, '200_ma']:
                if (df.at[i - 1, 'rsi'] + df.at[i, 'rsi']) < 10:
                    df.at[i, signal_direction_column] = 'positive'
                    df.at[i, signal_type_column] = 'cumulative_rsi'
    return df


def indicators_mid_levels_signal(stock_df, signal_direction_column, signal_type_column):
    df = stock_df.copy()
    for i in range(len(df)):
        if i > 1:
            if df['rsi'][i] > 50 and df['+di'][i] > 25 and df['-di'][i] < 25 and df['stochastic_k'][i] > 50:
                df.at[i, signal_direction_column] = 'positive'
                df.at[i, signal_type_column] = 'indicators_mid_levels_zone'
            # elif df['rsi'][i] < 50 and df['+di'][i] < 25 and df['-di'][i] > 25 and df['stochastic_k'][i] < 50:
            #     df[signal_direction_column][i] = 'negative'
            #     df[signal_type_column][i] = 'indicators_mid_levels_zone'
    return df


def joint_signal(stock_df, signal_direction_column, signal_type_column):
    df = stock_df.copy()
    for i in range(len(df)):
        if i > 1:
            if df['indicators_mid_level_direction'][i] == 'positive' and df['cross_50_direction'][i] == 'positive':
                df.at[i, signal_direction_column] = 'positive'
                df.at[i, signal_type_column] = 'joint_50'
            # elif df['indicators_mid_level_direction'][i] == 'negative' and df['cross_50_direction'][i] == 'negative':
            #     df[signal_direction_column][i] = 'negative'
            #     df[signal_type_column][i] = 'joint_50'
            elif df['indicators_mid_level_direction'][i] == 'positive' and df['cross_20_direction'][i] == 'positive':
                df.at[i, signal_direction_column] = 'positive'
                df.at[i, signal_type_column] = 'joint_20'
            # elif df['indicators_mid_level_direction'][i] == 'negative' and df['cross_20_direction'][i] == 'negative':
            #     df[signal_direction_column][i] = 'negative'
            #     df[signal_type_column][i] = 'joint_20'
    return df


def macd_cross_0_signal(stock_df, signal_direction_column, signal_type_column):
    df = stock_df.copy()
    for i in range(len(df)):
        if i > 1:
            if df['macd'][i] > 0 and df['macd'][i - 1] < 0:
                df[signal_direction_column][i] = 'positive'
                df[signal_type_column][i] = 'macd_cross_0'
            elif df['macd'][i] < 0 and df['macd'][i - 1] > 0:
                df[signal_direction_column][i] = 'negative'
                df[signal_type_column][i] = 'macd_cross_0'
    return df


def macd_signal_cross_signal(stock_df, signal_direction_column, signal_type_column):
    df = stock_df.copy()
    for i in range(len(df)):
        if i > 1:
            if df['macd'][i] > df['macd_signal'][i] and df['macd'][i - 1] < df['macd_signal'][i - 1]:
                df[signal_direction_column][i] = 'positive'
                df[signal_type_column][i] = 'macd_signal_cross'
            elif df['macd'][i] < df['macd_signal'][i] and df['macd'][i - 1] > df['macd_signal'][i - 1]:
                df[signal_direction_column][i] = 'negative'
                df[signal_type_column][i] = 'macd_signal_cross'
    return df


def joint_macd_signal_cross_signal(stock_df, signal_direction_column, signal_type_column):
    df = stock_df.copy()
    for i in range(len(df)):
        if i > 1:
            if df['indicators_mid_level_direction'][i] == 'positive' and df['macd_signal_cross_direction'][i] == 'positive':
                df[signal_direction_column][i] = 'positive'
                df[signal_type_column][i] = 'joint_macd_signal_cross'
            elif df['indicators_mid_level_direction'][i] == 'negative' and df['macd_signal_cross_direction'][i] == 'negative':
                df[signal_direction_column][i] = 'negative'
                df[signal_type_column][i] = 'joint_macd_signal_cross'
    return df


def joint_macd_cross_0_signal(stock_df, signal_direction_column, signal_type_column):
    df = stock_df.copy()
    for i in range(len(df)):
        if i > 1:
            if df['indicators_mid_level_direction'][i] == 'positive' and df['macd_cross_0_direction'][i] == 'positive':
                df[signal_direction_column][i] = 'positive'
                df[signal_type_column][i] = 'joint_macd_cross_0'
            elif df['indicators_mid_level_direction'][i] == 'negative' and df['macd_cross_0_direction'][i] == 'negative':
                df[signal_direction_column][i] = 'negative'
                df[signal_type_column][i] = 'joint_macd_cross_0'
    return df


def parabolic_trending_n_periods(stock_df, n, signal_direction_column, signal_type_column):
    # assuming parabolic trend (consistent divergence of price from 10 day moving average) will reverse
    # TODO: should be generic for n
    df = stock_df.copy()
    for i in range(len(df)):
        if i < n + 1:
            continue
        if df["distance_from_10_ma"][i-3] >= 0 and df["distance_from_10_ma"][i] <= 0.03 \
                and df["distance_from_10_ma"][i] >= df["distance_from_10_ma"][i-1] \
                and df["distance_from_10_ma"][i-1] >= df["distance_from_10_ma"][i-2] \
                and df["distance_from_10_ma"][i-2] >= df["distance_from_10_ma"][i-3]:
            df[signal_direction_column][i] = "negative"
            df[signal_type_column][i] = "parabolic_trend"
        elif df["distance_from_10_ma"][i-3] <= 0 and df["distance_from_10_ma"][i] >= -0.03 \
                and df["distance_from_10_ma"][i] <= df["distance_from_10_ma"][i-1] \
                and df["distance_from_10_ma"][i-1] <= df["distance_from_10_ma"][i-2] \
                and df["distance_from_10_ma"][i-2] <= df["distance_from_10_ma"][i-3]:
            df[signal_direction_column][i] = "positive"
            df[signal_type_column][i] = "parabolic_trend"
    return df


def check_volume_high_enough(df, i):
    return df['ma_volume'][i] != '' and df['Volume'][i] > df['ma_volume'][i] and df['Volume'][i] >= 1000000


def check_additional_positive_indicators(df, i):
    # return df['atr_volatility_ma'][i] > 0.03 and df['atr_volatility_ma'][i] < 0.09 and df['distance_from_10_ma'][i] > 0.04
    return df['atr_volatility_ma'][i] > 0.03 and df['atr_volatility_ma'][i] < 0.09 and df['distance_from_10_ma'][i] > -0.01 and df['distance_from_10_ma'][i] < 0.06 and df['adx'][i] > 12 and df['adx'][i] < 22


def check_atr_volatility_low_enough(df, i):
    return df['atr_volatility_ma'][i] != '' and df['atr_volatility_ma'][i] < 0.05


def check_not_earnings_days(df, i):
    return df['is_earning_days'][i] != True


def check_trend_not_down(df, i):
    return check_column_trend(df, 'stochastic_k', i) != 'DOWN'


def check_trend_not_up(df, i):
    return check_column_trend(df, 'stochastic_k', i) != 'UP'


def calculate_correl_score_series_for_df(df, correls_dict):
    # for i in range(len(df)):
    #     if df.at[i, 'signal_type'] == 'awesome_osc':
    #         df.at[i, 'position_score'] = df.at[i, 'median_ratio_norm']*0.0992+df.at[i, 'awesome_osc_norm']*0.0833+df.at[i, 'adx_norm']*-0.0598+df.at[i, 'atr_volatility_norm']*0.1443
    #     elif df.at[i, 'signal_type'] == 'joint_20' or df.at[i, 'signal_type'] == 'joint_50':
    #         df.at[i, 'position_score'] = df.at[i, 'median_ratio_norm']*0.054+df.at[i, 'atr_volatility_ma_norm']*-0.0417+df.at[i, 'adx_norm']*0.0465+df.at[i, 'adx_ma_med_5_rat']*-0.0878
    #     elif df.at[i, 'signal_type'] == 'cumulative_rsi':
    #         df.at[i, 'position_score'] = df.at[i, 'ma_med_5_ratio_norm']*0.0922
    # # TODO: I should normalize all position scores to be on a scale of -1 to 1 between themselves, so that they are on the same scale between the different signals, and not having one signal that is always greater only because it has stronger correlation with one or two features
    # # TODO: so separate the position_score series by signals, normalize -1 to 1 for each signal series separately, and then join together.
    # df_awesome_normalized_position_score = normalize_columns(df[df['signal_type'] == 'awesome_osc'], ['position_score'])
    # # df_joint_20_normalized_position_score = normalize_columns(df[df['signal_type'] == 'joint_20'], ['position_score'])
    # # df_joint_50_normalized_position_score = normalize_columns(df[df['signal_type'] == 'joint_50'], ['position_score'])
    # # df_cumulative_rsi_normalized_position_score = normalize_columns(df[df['signal_type'] == 'cumulative_rsi'], ['position_score'])
    # # df_merged = pd.concat([df_awesome_normalized_position_score, df_joint_20_normalized_position_score, df_joint_50_normalized_position_score, df_cumulative_rsi_normalized_position_score])
    # df_merged = pd.concat([df_awesome_normalized_position_score])

    for key, value in correls_dict.items():
        if df['position_score'].any() != '':
            df['position_score'] = df[key] * value + df['position_score']
        else:
            df['position_score'] = df[key] * value
    return df

