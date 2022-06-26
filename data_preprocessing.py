import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn


def get_timeseries_from_dataframe(df):
    """
    Transform the dataframe into 3 Time Series:
    - Quantity
    - MarketSilver
    - Month & day
    """
    qty_columns = [col for col in df.columns if col[:3] == 'qty']
    mktslvr_columns = [col for col in df.columns if col[:7] == 'mktslvr']
    months = df['month']
    mktslvr_ts = np.array([df[df['month'] == month][mktslvr_columns].values[0] for month in months]).flatten()
    qty_ts = np.array([df[df['month'] == month][qty_columns].values[0] for month in months]).flatten()
    month_ts = []
    for month in months:
        month_ts.extend([str(month) + '-' + col[7:] for col in mktslvr_columns])
    return mktslvr_ts, qty_ts, month_ts

def preprocess_timeseries(ts):
    """
    Pre-processes a single time-series to:
    - Interpolate missing values
    - Convert to pctl change
    - Remove trailing NaN values
    - Min Max scale the series
    """
    # Interpolate NaNs
    interpolated_ts = pd.Series(ts).interpolate()
    # Percent change
    pct_change = interpolated_ts.pct_change()
    # Remove trailing NaNs (TODO : retirer les mêmes valeurs partout)
    pct_change = pct_change.dropna(axis = 0)
    # Normalize
    return MinMaxScaler().fit_transform(pct_change.values.reshape(-1, 1)).flatten()
    
def train_test_split(ts):
    """
    Splits the Time Series into two sets.
    """
    return ts[:- len(ts) // 5], ts[len(ts) // 5:]

def create_inout_sequences(input_data, tw):
    """
    Creates input data for a PyTorch model.
    """
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

def multi_timeseries_uniformization(df_dict, columns):
    """
    Uniformization of all dataframes contained in the dictionary.
    """
    import pandas as pd
    subdfs = []
    for key, dataframe in df_dict.items():
        # Keep only asked columns
        subdf = dataframe.set_index('when')[columns]
        subdf.index = pd.to_datetime(subdf.index)
        # Renaming columns to keep the id name
        subdf.columns = [col + '_' + str(key) for col in columns]
        subdfs.append(subdf)
    merged_df = pd.concat(subdfs, axis = 1)
    return merged_df.interpolate()

def multi_timeseries_preprocessing(merged_df):
    """
    Applies preprocessing to all columns of a dataframe
    """
    new_columns = {}
    for col in merged_df.columns:
        ts = merged_df[col].values
        new_columns[col] = preprocess_timeseries(ts)
    return pd.DataFrame(new_columns)