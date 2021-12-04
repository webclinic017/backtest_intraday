#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from stocktrends import Renko
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler


def get_ATR_column_for_stock(stock_df, period, only_column=True):
    "function to calculate True Range and Average True Range"
    df = stock_df.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(period).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    if only_column == True:
        return df2['ATR']
    return df2


def get_volatility_from_atr(stock_df, average_periods):
    "function to calculate True Range and Average True Range"
    df = stock_df.copy()
    df['atr_volatility'] = df['atr'] / df['Close']
    df['atr_volatility_average'] = df['atr_volatility'].rolling(average_periods).mean()
    return df['atr_volatility'], df['atr_volatility_average']


def get_obv_column_for_stock(stock_df):
    """function to calculate On Balance Volume"""
    df = stock_df.copy()
    df['daily_ret'] = df['Close'].pct_change()
    df['direction'] = np.where(df['daily_ret']>=0,1,-1)
    df['direction'][0] = 0
    df['vol_adj'] = df['Volume'] * df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df['obv']

def get_macd_columns_for_stock(stock_df,a,b,c):
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""
    df = stock_df.copy()
    df["MA_Fast"]=df["Close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["Close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    # df.dropna(inplace=True)
    return df["MACD"], df["Signal"]

def get_rolling_max_column_for_stock(stock_df, column_name, period):
    df = stock_df.copy()
    df[f'roll_20_max_{column_name}'] = stock_df[column_name].rolling(20).max()
    return df[f'roll_20_max_{column_name}']

def get_ma_column_for_stock(stock_df, column_name, period):
    df = stock_df.copy()
    df[f'ma_{period}'] = stock_df[column_name].rolling(period).mean()
    return df[f'ma_{period}']

def get_distance_between_columns_for_stock(stock_df, orig_column_name, to_column_name):
    df = stock_df.copy()
    df['pct_change_between_columns'] = (df[orig_column_name] - df[to_column_name]) / df[orig_column_name]
    return df['pct_change_between_columns']

def get_rolling_min_column_for_stock(stock_df, column_name, period):
    df = stock_df.copy()
    df[f'roll_20_min_{column_name}'] = stock_df[column_name].rolling(20).min()
    return df[f'roll_20_min_{column_name}']

def get_basic_renko(stock_df, period):
    "function to convert ohlc data into renko bricks"
    df = stock_df.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:,[0,1,2,3,4,5]]
    df.columns = ["date","open","high","low","close","volume"]
    df2 = Renko(df)
    # completely not sure about this renko brick calculation
    atr_df = get_ATR_column_for_stock(stock_df, 120, False)
    df2.brick_size = max(0.5, round(atr_df["ATR"][-1],0))
    print(f'brick size: {df2.brick_size}')
    renko_df = df2.get_ohlc_data()
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
    renko_df.columns = ["Date","open","high","low","close","uptrend","bar_num"]
    df_to_merge = stock_df.copy()
    df_to_merge["Date"] = df_to_merge.index
    df_to_merge = df_to_merge.merge(renko_df.loc[:,["Date","bar_num"]],how="outer",on="Date")
    df_to_merge.set_index('Date', inplace=True)
    df_to_merge["bar_num"].fillna(method='ffill',inplace=True)
    return df_to_merge['bar_num']

def slope(series, period):
    "function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(period-1)]
    for i in range(period,len(series)+1):
        y = series[i-period:i]
        x = np.array(range(period))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

def BollBnd(DF,n):
    "function to calculate Bollinger Band"
    df = DF.copy()
    df["MA"] = df['Close'].rolling(n).mean()
    df["BB_up"] = df["MA"] + 2*df['Close'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_dn"] = df["MA"] - 2*df['Close'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_width"] = df["BB_up"] - df["BB_dn"]
    return df['BB_up'], df['BB_dn'], df['BB_width']

def get_adx_column_for_stock(stock_df, n):
    "function to calculate ADX"
    """0-25	Absent or Weak Trend
    25-50	Strong Trend
    50-75	Very Strong Trend
    75-100	Extremely Strong Trend"""
    df2 = stock_df.copy()
    df2['TR'] = get_ATR_column_for_stock(df2, n, False)['TR'] #the period parameter of ATR function does not matter because period does not influence TR calculation
    df2['DMplus']=np.where((df2['High']-df2['High'].shift(1))>(df2['Low'].shift(1)-df2['Low']),df2['High']-df2['High'].shift(1),0)
    df2['DMplus']=np.where(df2['DMplus']<0,0,df2['DMplus'])
    df2['DMminus']=np.where((df2['Low'].shift(1)-df2['Low'])>(df2['High']-df2['High'].shift(1)),df2['Low'].shift(1)-df2['Low'],0)
    df2['DMminus']=np.where(df2['DMminus']<0,0,df2['DMminus'])
    TRn = []
    DMplusN = []
    DMminusN = []
    TR = df2['TR'].tolist()
    DMplus = df2['DMplus'].tolist()
    DMminus = df2['DMminus'].tolist()
    for i in range(len(df2)):
        if i < n:
            TRn.append(np.NaN)
            DMplusN.append(np.NaN)
            DMminusN.append(np.NaN)
        elif i == n:
            TRn.append(df2['TR'].rolling(n).sum().tolist()[n])
            DMplusN.append(df2['DMplus'].rolling(n).sum().tolist()[n])
            DMminusN.append(df2['DMminus'].rolling(n).sum().tolist()[n])
        elif i > n:
            TRn.append(TRn[i-1] - (TRn[i-1]/n) + TR[i])
            DMplusN.append(DMplusN[i-1] - (DMplusN[i-1]/n) + DMplus[i])
            DMminusN.append(DMminusN[i-1] - (DMminusN[i-1]/n) + DMminus[i])
    df2['TRn'] = np.array(TRn)
    df2['DMplusN'] = np.array(DMplusN)
    df2['DMminusN'] = np.array(DMminusN)
    df2['DIplusN']=100*(df2['DMplusN']/df2['TRn'])
    df2['DIminusN']=100*(df2['DMminusN']/df2['TRn'])
    df2['DIdiff']=abs(df2['DIplusN']-df2['DIminusN'])
    df2['DIsum']=df2['DIplusN']+df2['DIminusN']
    df2['DX']=100*(df2['DIdiff']/df2['DIsum'])
    ADX = []
    DX = df2['DX'].tolist()
    for j in range(len(df2)):
        if j < 2*n-1:
            ADX.append(np.NaN)
        elif j == 2*n-1:
            ADX.append(df2['DX'][j-n+1:j+1].mean())
        elif j > 2*n-1:
            ADX.append(((n-1)*ADX[j-1] + DX[j])/n)
    df2['ADX']=np.array(ADX)
    return df2['ADX'], df2['DIplusN'], df2['DIminusN']

def rsi(stock_df, n):
    "function to calculate RSI"
    df = stock_df.copy()
    delta = df['Close'].diff().dropna()
    d_up, d_down = delta.copy(), delta.copy()
    d_up[d_up < 0] = 0
    d_down[d_down > 0] = 0
    rol_up = d_up.rolling(n).mean()
    rol_down = d_down.rolling(n).mean().abs()
    rs = rol_up / rol_down
    rsi_col = 100.0 - (100.0 / (1.0 + rs))
    return rsi_col


def stochastic(stock_df,periods,smother):
    "function to calculate stochastic"
    df = stock_df.copy()
    df['14-high'] = df['High'].rolling(periods).max()
    df['14-low'] = df['Low'].rolling(periods).min()
    df['%K'] = (df['Close'] - df['14-low']) * 100 / (df['14-high'] - df['14-low'])
    df['%D'] = df['%K'].rolling(smother).mean()
    return df['%K'], df['%D']


def roll(df, w):
    for i in range(df.shape[0] - w + 1):
        yield pd.DataFrame(df.values[i:i+w, :], df.index[i:i+w], df.columns)


def beta(df, benchmark_col_name):
    market = df[benchmark_col_name]
    X = market.values.reshape(-1, 1)
    X = np.concatenate([np.ones_like(X), X], axis=1)
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values)
    return pd.Series(b[1], df.columns, name=df.index[-1])


def get_beta_column(stock_df, benchmark_df, period):
    df = pd.concat([benchmark_df['Close'], stock_df['Close']], axis=1, keys=['benchmark_close', 'Close'])
    betas = pd.concat([beta(sdf, 'benchmark_close') for sdf in roll(df.pct_change().dropna(), period)], axis=1).T
    return betas['Close']


def normalize_columns(df, columns_list):
    temp_df = df.copy()
    for col in columns_list:
        scaler = MinMaxScaler((-1, 1))
        temp_df[f'temp_{col}'] = df[col].clip(df[col].quantile(.05), df[col].quantile(.95))
        temp_df[[col]] = scaler.fit_transform(temp_df[[f'temp_{col}']])
        df[f'{col}_norm'] = temp_df[col]
    return df
