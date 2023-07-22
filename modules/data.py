from typing import Tuple
import pandas as pd
import pyreadr

from datetime import datetime, timezone
from .config import IST
#comment
def get_df_from_rds(path: str, localize=True) -> pd.DataFrame:
    '''
    function to convert rds file to dataframe
    '''
    df = pyreadr.read_r(path)
    df = df[None]
    
    if localize:
        df.ts = df.ts.apply(lambda x: x.tz_localize('utc'))
        df.ts = df.ts.apply(lambda x: x.astimezone(IST))
        return df

    return df

def localize_df(df) -> pd.DataFrame:
    '''
    function to convert rds file to dataframe
    '''
    # df = pyreadr.read_r(path)
    # df = df[None]
    
    df['ts'] = pd.to_datetime(df['ts'])

    # print(df)
    
    # df.ts = df.ts.apply(lambda x: x.tz_localize('utc'))
    try:
        df.ts = df.ts.apply(lambda x: x.astimezone(IST))
        # print(df)
    except:
        df.ts = df.ts.apply(lambda x: x.tz_localize(IST))
        # print(df)
        # df.ts = df.ts.apply(lambda x: x.astimezone(IST))
        # print(df)
    
    return df

def todatetime_df(df, cols) -> pd.DataFrame:
    '''
    function to convert rds file to dataframe
    '''
    # df = pyreadr.read_r(path)
    # df = df[None]

    # print(df)
    
    for col in cols:
        df[col] = pd.to_datetime(df[col])
        df[col] = df[col].apply(lambda x: x.tz_localize('utc'))
        df[col+'_IST'] = df[col].apply(lambda x: x.astimezone(IST))
    return df

def get_data_range(df: pd.DataFrame, start_date: str, end_date:str, ts_col='ts') -> pd.DataFrame:
    '''
    function to get data between a particular range
    '''
    d1 = [int(i) for i in start_date.split('-')]
    d2 = [int(i) for i in end_date.split('-')]

    d1 = datetime(d1[0], d1[1], d1[2], 0, 0, 0, 0, IST)
    d2 = datetime(d2[0], d2[1], d2[2], 23, 59, 59, 0, IST)

    df_range = df[(df[ts_col]>=d1) & (df[ts_col]<=d2)]

    try:
        return df_range.reset_index(drop=True)
    except:
        return df_range


def filter_by_topic_param(df, topic:list, param:int, topic_col='topic', param_col='ST') -> pd.DataFrame:
    '''
    function to get data for a particular topic (site + meter) and register
    '''
    try:
        df_filtered = df[(df[topic_col].isin(topic)) & (df[param_col]==param)].drop(['index', 'TS'], axis=1)
    except Exception as e:
        df_filtered = df[(df[topic_col].isin(topic)) & (df[param_col]==param)]
        print(f"\nError filtering: {e}, fetching df...")

    try:
        return df_filtered.reset_index(drop=True)
    except:
        return df_filtered

def filter_by_topic(df, topic:str, topic_col='veh') -> pd.DataFrame:
    '''
    function to get data for a particular topic (site + meter) and register
    '''
    # try:
    #     df_filtered = df[df[topic_col].isin(topic)].drop(['index', 'TS'], axis=1)
    # except Exception as e:
    df_filtered = df[df[topic_col]==topic]
        # print(f"\nError filtering: {e}, fetching df...")

    try:
        return df_filtered.reset_index(drop=True)
    except:
        return df_filtered
