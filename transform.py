# This file is credited to CaQtiml
# https://github.com/CaQtiml

import pandas as pd


# calculate moving average of the last x hours
# window_size determine hours
def df_ma_x_hour(inp_df, window_size, col_name, round_decimal=10, idx_name='time'):
    df_ma_x_h_p = inp_df.groupby(inp_df.index.minute).rolling(window=window_size)[col_name].mean().round(round_decimal)
    df_ma_x_h = df_ma_x_h_p.droplevel(0).reset_index().set_index(idx_name).sort_index()
    return df_ma_x_h

# calculate moving average of the last x days
# window_size determine days
def df_ma_x_day(inp_df, window_size, col_name, round_decimal=10, idx_name='time'):
    df_ma_x_d_p = inp_df.groupby([inp_df.index.hour, inp_df.index.minute]).rolling(window=window_size)[col_name].mean().round(round_decimal)
    df_ma_x_d = df_ma_x_d_p.droplevel([0,1]).reset_index().set_index(idx_name).sort_index()
    return df_ma_x_d

def diff_ma(df_one, df_two, col_rename, round_decimal=10):
    return (df_one - df_two).round(round_decimal).rename(columns=col_rename)

def window_normalization(inp_df, window_size, col_name, round_decimal=10):
    rolling_min = inp_df[col_name].rolling(window=window_size).min()
    rolling_max = inp_df[col_name].rolling(window=window_size).max()

    # Calculate the last value in the rolling window
    last_value = inp_df[col_name].rolling(window=window_size).apply(lambda x: x.iloc[-1])

    # Apply min-max scaling using pre-computed rolling min and max
    scaled_df = ((last_value - rolling_min) / (rolling_max - rolling_min)).round(round_decimal)
    return scaled_df

def transform(df):
    df_close = df[['close', 'currency']]
    
    # Calculate ma of 7h, 25h, 99h
    df_ma7h = df_ma_x_hour(inp_df=df, window_size=7, col_name='close', idx_name='time')
    df_ma25h = df_ma_x_hour(inp_df=df, window_size=25, col_name='close', idx_name='time')
    df_ma99h = df_ma_x_hour(inp_df=df, window_size=99, col_name='close', idx_name='time')
    
    # Transform indicator ma7_25h, ma25_99h
    df_ma7_25h = diff_ma(df_one=df_ma7h, df_two=df_ma25h, col_rename={'close': 'ma7_25h'})
    df_ma25_99h = diff_ma(df_one=df_ma25h, df_two=df_ma99h, col_rename={'close': 'ma25_99h'})
    
    # Calculate ma of 7d, 25d
    df_ma7d = df_ma_x_day(inp_df=df, window_size=7, col_name='close', idx_name='time')
    df_ma25d = df_ma_x_day(inp_df=df, window_size=25, col_name='close', idx_name='time')
    
    # Transform indicator ma7_25d
    df_ma7_25d = diff_ma(df_one=df_ma7d, df_two=df_ma25d, col_rename={'close': 'ma7_25d'})
    
    # Merge all indicators and close price
    final_df = pd.merge(df_ma7_25h, df_ma7_25d, how='outer', left_index=True, right_index=True, suffixes=('_df1', '_df2'))
    final_df = pd.merge(df_ma25_99h, final_df, how='outer', left_index=True, right_index=True, suffixes=('_df1', '_df2'))
    final_df = pd.merge(df_close, final_df, how='outer', left_index=True, right_index=True, suffixes=('_df1', '_df2'))

    # Scale the indicators
    final_df.dropna()
    final_df["ma25_99h_scale"] = (final_df["ma25_99h"]/final_df["close"]).round(10)
    final_df["ma7_25h_scale"] = (final_df["ma7_25h"]/final_df["close"]).round(10)
    final_df["ma7_25d_scale"] = (final_df["ma7_25d"]/final_df["close"]).round(10)
    final_df.dropna()

    # Scale min-max close price, then merge into the final_df
    scaled_df = window_normalization(inp_df=df, window_size=37500, col_name='close')
    scaled_df_clean = scaled_df.to_frame().rename(columns={'close': 'close_minmax_scale'})
    final_df = pd.merge(scaled_df_clean, final_df, how='outer', left_index=True, right_index=True, suffixes=('_df1', '_df2'))

    return final_df


