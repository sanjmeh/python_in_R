import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from haversine import haversine, Unit
from tqdm import tqdm
import pytz
import math
import os
# print('import successful')


def timestamp_(date):
    formatted_datetime = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime

def Utc_to_Ist(utc_time1):

    utc_time = datetime.strptime(str(utc_time1), "%Y-%m-%d %H:%M:%S")
    ist_timezone = pytz.timezone("Asia/Kolkata")
    ist_time = utc_time.replace(tzinfo=pytz.UTC).astimezone(ist_timezone).strftime("%Y-%m-%d %H:%M:%S")
    return ist_time
# print('function done')



def data_prep_distance(df,mods_df):

    df = df.reset_index(drop=True)
    df['ts'] = pd.to_datetime(df['ts'])
    df['date'] = [str(i).split(' ')[0] for i in df['ts']]
    start_time = pd.to_datetime(df.head(1)['date'].item()).replace(hour=22, minute=0, second=0)
    end_time = pd.to_datetime(df.tail(1)['date'].item()).replace(hour=22, minute=0, second=0) 
    if df[df['ts'] >= end_time].empty:
        end_time -= pd.DateOffset(days=1)
    df=df[(df['ts'] >= start_time) & (df['ts'] <= end_time)]  

    if len(mods_df)!=0:
        mods_df['start_date'] = [str(i).split(' ')[0] for i in mods_df['strt']]
        mods_df['end_date'] = [str(i).split(' ')[0] for i in mods_df['end']]
        start_time_mod = pd.to_datetime(mods_df.head(1)['start_date'].item()).replace(hour=22, minute=0, second=0)
        end_time_mod = pd.to_datetime(mods_df.tail(1)['end_date'].item()).replace(hour=22, minute=0, second=0)
        mods_df['strt'] = pd.to_datetime(mods_df['strt'])
        mods_df['end'] = pd.to_datetime(mods_df['end'])
        if mods_df[mods_df['end'] >= end_time_mod].empty:
            end_time_mod -= pd.DateOffset(days=1)
        mods_df=mods_df[(mods_df['strt'] >= start_time_mod) & (mods_df['end'] <= end_time_mod)]
        mods_df1 = mods_df.query("fuel>100")
        if len(mods_df1)==0:
            mods_df1 = mods_df.query("fuel>60")
    else:
        pass

    df['cumsum_dist'] = df['Haversine_dist'].cumsum()
    df['cumsum_dist_Sir'] = df['disthav'].cumsum()
    df.rename(columns={'currentFuelVolumeTank1':'fuel'},inplace=True)
    diff = df['fuel'].diff().fillna(0)
    df['fuel_diff'] = np.where(diff == 0, 0, -diff)
    df['ts'] = pd.to_datetime(df['ts'])
    df['time_diff'] = df['ts'].diff().fillna(pd.Timedelta(minutes=0)).dt.total_seconds() / 60
#     df['time_diff'] = df['time_diff'].fillna(0)
    df['lph'] = (df['fuel_diff']/df['time_diff'])*60
    df['lp100km'] = ((df['fuel_diff']/df['Haversine_dist'])*1000)*100
    df=df.reset_index(drop=True)    
    
    df['REfuel_unique'] = 0
    if len(mods_df)!=0:
        for _,row in mods_df.iterrows():
            sample=df.loc[(df['ts']>=row['strt'])&(df['ts']<=row['end'])]
            if len(sample)!=0:
                df.loc[sample.index[0], 'REfuel_unique'] = row['fuel']
    else:
        pass

    return df, mods_df
# print('dist function done')


def data_prep_hour(df,mods_df):
    
    df['ts'] = pd.to_datetime(df['ts'])
    df['date'] = [str(i).split(' ')[0] for i in df['ts']]
    start_time = pd.to_datetime(df.head(1)['date'].item()).replace(hour=22, minute=0, second=0)
    end_time = pd.to_datetime(df.tail(1)['date'].item()).replace(hour=22, minute=0, second=0)
    if df[df['ts'] >= end_time].empty:
        end_time -= pd.DateOffset(days=1)
    df=df[(df['ts'] >= start_time) & (df['ts'] <= end_time)]

    if len(mods_df)!=0:
        mods_df['start_date'] = [str(i).split(' ')[0] for i in mods_df['strt']]
        mods_df['end_date'] = [str(i).split(' ')[0] for i in mods_df['end']]
        start_time_mod = pd.to_datetime(mods_df.head(1)['start_date'].item()).replace(hour=22, minute=0, second=0)
        end_time_mod = pd.to_datetime(mods_df.tail(1)['end_date'].item()).replace(hour=22, minute=0, second=0)
        mods_df['strt'] = pd.to_datetime(mods_df['strt'])
        mods_df['end'] = pd.to_datetime(mods_df['end'])
        if mods_df[mods_df['end'] >= end_time_mod].empty:
            end_time_mod -= pd.DateOffset(days=1)
        mods_df=mods_df[(mods_df['strt'] >= start_time_mod) & (mods_df['end'] <= end_time_mod)]
        mods_df1 = mods_df.query("fuel>100")
        if len(mods_df1)==0:
            mods_df1 = mods_df.query("fuel>60")
    else:
        pass

    df['cumsum_dist'] = df['Haversine_dist'].cumsum()
    df['cumsum_dist_Sir'] = df['disthav'].cumsum()
    df.rename(columns={'currentFuelVolumeTank1':'fuel'},inplace=True)
    diff = df['fuel'].diff().fillna(0)
    df['fuel_diff'] = np.where(diff == 0, 0, -diff)
    df['ts'] = pd.to_datetime(df['ts'])
    df['time_diff'] = df['ts'].diff().fillna(pd.Timedelta(minutes=0)).dt.total_seconds() / 60
    df['Cum_Timediff'] = df['time_diff'].cumsum().fillna(0)
#     df.loc[0, 'Cum_Timediff'] = df.loc[0, 'time_diff']
#     df['time_diff'] = df['time_diff'].fillna(0)
    df['lph'] = (df['fuel_diff']/df['time_diff'])*60
    df['lp100km'] = ((df['fuel_diff']/df['Haversine_dist'])*1000)*100
    df=df.reset_index(drop=True)    
    
    df['REfuel_unique'] = 0
    if len(mods_df)!=0:
        for _,row in mods_df.iterrows():
            sample=df.loc[(df['ts']>=row['strt'])&(df['ts']<=row['end'])]
            if len(sample)!=0:
                df.loc[sample.index[0], 'REfuel_unique'] = row['fuel']
    else:
        pass

    return df, mods_df
# print('hour function worked')


def data_prep_fuel(df,mods_df):

    df['ts'] = pd.to_datetime(df['ts'])
    df['date'] = [str(i).split(' ')[0] for i in df['ts']]
    start_time = pd.to_datetime(df.head(1)['date'].item()).replace(hour=22, minute=0, second=0)
    end_time = pd.to_datetime(df.tail(1)['date'].item()).replace(hour=22, minute=0, second=0)
    if df[df['ts'] >= end_time].empty:
        end_time -= pd.DateOffset(days=1)
    df=df[(df['ts'] >= start_time) & (df['ts'] <= end_time)]
#     print('S3')
    

    if len(mods_df)==0:
#         mods_df = allmod_df[allmod_df['termid']==term_id].query("fuel>60")
#     else:
        pass
#         if len(mods_df)==0:
#             mods_df = None
#         else:
#             pass
    if len(mods_df)!=0:
        mods_df['start_date'] = [str(i).split(' ')[0] for i in mods_df['strt']]
        mods_df['end_date'] = [str(i).split(' ')[0] for i in mods_df['end']]
        start_time_mod = pd.to_datetime(mods_df.head(1)['start_date'].item()).replace(hour=22, minute=0, second=0)
        end_time_mod = pd.to_datetime(mods_df.tail(1)['end_date'].item()).replace(hour=22, minute=0, second=0)
        mods_df['strt'] = pd.to_datetime(mods_df['strt'])
        mods_df['end'] = pd.to_datetime(mods_df['end'])
        if mods_df[mods_df['end'] >= end_time_mod].empty:
            end_time_mod -= pd.DateOffset(days=1)
        mods_df=mods_df[(mods_df['strt'] >= start_time_mod) & (mods_df['end'] <= end_time_mod)]
        mods_df=mods_df.query("fuel>20")
    else:
        pass

    df['REfuel_amt'] = 0
    for _,row in mods_df.iterrows():
        df.loc[(df['ts']>=row['strt'])&(df['ts']<=row['end']),'REfuel_amt'] = row['fuel']
# 
    diff = df['currentFuelVolumeTank1'].diff().fillna(0)
    df['Fuel_difference'] = np.where(diff == 0, 0, -diff)
    
    sample_set = set()
    for index,row in df.query("REfuel_amt>0").iterrows():
        sample_set.add(index)
        sample_set.add(index+1)
    indexes = list(sample_set)
    df.loc[indexes,'Fuel_difference'] = 0

    s=0;t=0
    for index,row in df.iterrows():
        s+=row['Fuel_difference']
        if s >110:
            s = row['Fuel_difference']
            t += 1
        df.loc[index,'Cum_Fuelcons'] = s
        df.loc[index,'Bucket'] = t
    df['cumsum_dist'] = df['Haversine_dist'].cumsum().fillna(0)
    df['ts'] = pd.to_datetime(df['ts'])
    df['time_diff'] = df['ts'].diff().dt.total_seconds() / 60
    df['time_diff'] = df['time_diff'].fillna(0)
    df['lph'] = (df['Fuel_difference']/df['time_diff'])*60
    df['lp100km'] = ((df['Fuel_difference']/df['Haversine_dist'])*1000)*100
    df=df.reset_index(drop=True) 

    return df,mods_df
# print('fuel function worked')





