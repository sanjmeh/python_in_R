import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from d_config import cst_data_path,ign_data_path,output_data_path
from datetime import datetime, timedelta, time
from haversine import haversine, Unit
from multiprocess import cpu_count
from p_tqdm import p_map
from tqdm import tqdm
from time import sleep
import pytz
import pyreadr
import math
import os
tqdm.pandas()
import warnings
warnings.filterwarnings("ignore")


def categorize_shift(hour):
    if 6 <= hour < 14:
        return 'A'
    elif 14 <= hour < 22:
        return 'B'
    else:
        return 'C'
    
def calculate_consecutive_haversine_distances(datam):
    distances = []
    for i in range(1, len(datam)):
        lat1, lon1 = datam.at[i-1, 'lt'], datam.at[i-1, 'lg']
        lat2, lon2 = datam.at[i, 'lt'], datam.at[i, 'lg']
        distance = haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)
        distances.append(distance)
    distances.insert(0,0)
    return distances

def continuous_position_wise_grouping(a):
    buckets = []
    start = 0
    is_zero_bucket = a[0] == 0
    for i, num in enumerate(a[1:], start=1):
        if num == 0 and not is_zero_bucket:
            end = i
            is_zero_bucket = True
            buckets.append((start, end))
            start = end  
        elif num != 0 and is_zero_bucket:
            end = i
            is_zero_bucket = False
            buckets.append((start, end))
            start = end  
    if is_zero_bucket:
        end = len(a)
        buckets.append((start, end))
    elif not is_zero_bucket and start != len(a):
        end = len(a)
        buckets.append((start, end))
    return buckets

def add_stationary_column(datam):
    datam['status'] = 'stationary'
    return datam
def add_movement_column(datam):
    datam['status'] = 'movement'
    return datam

def get_shift_timestamp(date_str):
    datetime_input = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    input_time = datetime_input.time()
    if input_time >= datetime.strptime('00:00:00', '%H:%M:%S').time() and input_time < datetime.strptime('06:00:00', '%H:%M:%S').time():
        shift_time = datetime_input.replace(hour=6, minute=0, second=0, microsecond=0)
    elif input_time >= datetime.strptime('06:00:00', '%H:%M:%S').time() and input_time < datetime.strptime('14:00:00', '%H:%M:%S').time():
        shift_time = datetime_input.replace(hour=14, minute=0, second=0, microsecond=0)
    elif input_time >= datetime.strptime('14:00:00', '%H:%M:%S').time() and input_time < datetime.strptime('22:00:00', '%H:%M:%S').time():
        shift_time = datetime_input.replace(hour=22, minute=0, second=0, microsecond=0)
    else:
        shift_time = (datetime_input + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
    return shift_time

def row_split(start_time,end_time):
    end = str(get_shift_timestamp(start_time))
    start_list=[];end_list=[]
    while pd.to_datetime(end)<pd.to_datetime(end_time):
        start_list.append((start_time,end))
        start_time = end
        end = str(get_shift_timestamp(start_time))
    else:
        start_list.append((start_time,end_time))
    return start_list

def fuel_interpolation(initial_level, end_level, increments_list,total_time):

    step_size = (end_level - initial_level) / total_time
    in_list = increments_list.copy()
    in_list.pop(-1)
    buckets = []
    for increment in in_list:
        bucket_time = (pd.to_datetime(increment[1])-pd.to_datetime(increment[0])).total_seconds()/60
        bucket_start = initial_level
        bucket_end = initial_level + (bucket_time * step_size)
        buckets.append((bucket_start, bucket_end))
        initial_level = bucket_end
    buckets.append((buckets[-1][1], end_level))
    return buckets

def ign_time_cst(a,b):
    # a = ignstatus column ;  b = Time difference column
    buckets = [];start_index = None
    for i, value in enumerate(a):
        if value == 1:
            if start_index is None:
                start_index = i
        elif start_index is not None:
            buckets.append((start_index, i - 1))
            start_index = None
    if start_index is not None:
        buckets.append((start_index, len(a) - 1))
    ign_time=0
    for j in buckets:
        s = sum(b[(j[0]+1):(j[1]+1)])
        try:
            s = s+(b[j[0]]/2)+(b[j[1]+1]/2)
        except:
            s=s+(b[j[0]]/2)
        ign_time=ign_time+s
    return ign_time

df = pd.read_csv(cst_data_path,parse_dates=['ts'], infer_datetime_format=True)
df['ts'] = df['ts'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
df['ts'] = pd.to_datetime(df['ts'])
df['date'] = df['ts'].dt.date.astype(str)
df['hour'] = df['ts'].dt.hour

termid_list = df['termid'].unique().tolist()

def dist_allmods(i):
    
    term_df = df[df['termid']==i]
    term_df=term_df.reset_index(drop=True)
    term_df['shift'] = term_df['hour'].apply(categorize_shift)
    term_df['Haversine_dist'] = calculate_consecutive_haversine_distances(term_df)
    term_df['Fuel_diff'] = term_df['currentFuelVolumeTank1'].diff().fillna(0)
    term_df.sort_values(by=['ts'],inplace=True)
    term_df['Time_diff'] = term_df['ts'].diff().fillna(pd.Timedelta(minutes=0)).dt.total_seconds() / 60
    term_df['Cons_Speed'] = term_df['Haversine_dist']/term_df['Time_diff']
    term_df['Cons_Speed'] = term_df['Cons_Speed'].fillna(0)
    term_df['status'] = 1
    term_df.loc[term_df['Cons_Speed']<50 , 'status'] = 0

    bucket = continuous_position_wise_grouping(term_df['status'].tolist())
    list_=[]
    for index,j in enumerate(bucket):
        if j[0]!=0:
            sample = term_df.iloc[j[0]-1:j[1]]       
        else:
            sample = term_df.iloc[j[0]:j[1]]    
            if len(sample)==1:
                try:
                    inc = bucket[index+1]
                    sample= term_df.iloc[j[0]:inc[1]]
                except:
                    pass
        sample = sample.reset_index(drop=True)
        sample['ts'] = pd.to_datetime(sample['ts'])
        sample['new_time_diff'] = sample['ts'].diff().fillna(pd.Timedelta(minutes=0)).dt.total_seconds() / 60
        start_d = sample.head(1)['date'].item();start_time=sample.head(1)['ts'].item()
        end_d = sample.tail(1)['date'].item();end_time=sample.tail(1)['ts'].item()
        sample['new_distance']= calculate_consecutive_haversine_distances(sample)
        ig_time = sample[sample['currentIgn']==1]
        start_level=sample.head(1)['currentFuelVolumeTank1'].item()
        end_level=sample.tail(1)['currentFuelVolumeTank1'].item()
        if start_d == end_d:
            date = 'Same'
        else:
            date = 'Different'
        start_shift = sample.head(1)['shift'].item();end_shift=sample.tail(1)['shift'].item()
        total_time = (end_time-start_time).total_seconds()/60
        if (start_shift==end_shift)&(((date=='Same')&((start_shift=='B')or(start_shift=='A')or((start_shift=='C')&(total_time<480))))or((date=='Different')&(start_shift=='C')&(total_time<480))):
            term_dict={}
            ign_cst = ign_time_cst(sample['currentIgn'].tolist(),sample['new_time_diff'].tolist())
            keys = ['termid','reg_numb','start_time','end_time','total_obs','start_lt','start_lg','end_lt','end_lg',
                    'max_time_gap','initial_level','end_level','total_dist','ign_perc','ign_time_cst']
            values = [[sample.head(1)['termid'].item()],[sample.head(1)['regNumb'].item()],[start_time],[end_time],[len(sample)],
                      [sample.head(1)['lt'].item()],[sample.head(1)['lg'].item()],[sample.tail(1)['lt'].item()],
                      [sample.tail(1)['lg'].item()],[sample['new_time_diff'].max()],[sample.head(1)['currentFuelVolumeTank1'].item()],
                      [sample.tail(1)['currentFuelVolumeTank1'].item()],[sample['new_distance'].sum()],[(len(ig_time)/len(sample))*100],
                     [ign_cst]]
            term_dict.update(zip(keys,values))
            within_df = pd.DataFrame(term_dict)
            within_df['Interpolation_status'] = 'Both_Real'
        else:
            sample_list = row_split(str(start_time),str(end_time))
            l=[];
            fuel_inter = fuel_interpolation(start_level,end_level,sample_list,total_time)
            for k in range(len(sample_list)):
                temp_dict={}
                keys2=['termid','reg_numb','start_time','end_time','total_obs','initial_level','end_level']
                values2=[i,sample.head(1)['regNumb'].item(),sample_list[k][0],sample_list[k][1],
                         len(sample[(sample['ts']>=pd.to_datetime(sample_list[k][0]))&(sample['ts']<=pd.to_datetime(sample_list[k][1]))]),
                         fuel_inter[k][0],fuel_inter[k][1]]                  
                temp_dict.update(zip(keys2,values2))
                l.append(temp_dict)
            within_df = pd.DataFrame(l)
            within_df['start_lt'] = sample.head(1)['lt'].item()
            within_df['start_lg'] = sample.head(1)['lg'].item()
            within_df['end_lt'] = sample.tail(1)['lt'].item()
            within_df['end_lg'] = sample.tail(1)['lg'].item()
            within_df = within_df.reset_index(drop=True)
            within_df.loc[0,'Interpolation_status'] = 'End_interpolated'
            within_df.loc[within_df.index[-1],'Interpolation_status'] = 'Start_interpolated'
        list_.append(within_df)
    list_[::2] = [add_stationary_column(df) for df in list_[::2]]
    list_[1::2] = [add_movement_column(df) for df in list_[1::2]]
    ff=pd.concat(list_)
    ff.loc[ff['Interpolation_status'].isnull()==True,'Interpolation_status']='Both_Interpolated'
    ff['start_time'] = pd.to_datetime(ff['start_time'])
    ff['end_time']=pd.to_datetime(ff['end_time'])
    ff.sort_values(by=['start_time'],inplace=True)
    ff['start_hour'] = ff['start_time'].dt.hour
    ff['end_hour'] = ff['end_time'].dt.hour
    ff['start_shift'] = ff['start_hour'].apply(categorize_shift)
    ff['end_shift'] = ff['end_hour'].apply(categorize_shift)
    sleep(.00001)
    
    return ff

ign = pd.read_csv(ign_data_path,parse_dates=['strt','end'], infer_datetime_format=True)
ign['strt'] = ign['strt'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
ign['end'] = ign['end'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
ign['strt'] = pd.to_datetime(ign['strt'])
ign['end'] = pd.to_datetime(ign['end'])

def ign_time_int(row):
    x_start = row['start_time']
    x_end = row['end_time']
    termid = row['termid']
    ign_ = ign.loc[(((ign['strt']<=x_end)&(ign['strt']>=x_start)) | ((ign['end']<=x_end)&(ign['end']>=x_start)) | ((ign['strt']<=x_start)&(ign['end']>=x_end)))&(ign['termid']==termid)]
    ign_.loc[ign_['strt']<x_start,'strt']=x_start
    ign_.loc[ign_['end']>x_end,'end']=x_end
    ign_['dur(mins)']=(ign_['end']-ign_['strt'])/timedelta(minutes=1)
    row['ign_time'] = sum(ign_['dur(mins)'])
    return row

def final_data_f(datam):
    datam[['start_time', 'end_time']] = datam[['start_time', 'end_time']].apply(pd.to_datetime)
    datam['total_cons']=datam['initial_level']-datam['end_level']
    datam['lp100k'] = datam.apply(lambda row: (row['total_cons']/row['total_dist'])*100000 if row['total_dist'] > 0 else 'NaN', axis=1)
    datam['total_time'] = (datam['end_time']-datam['start_time']).dt.total_seconds()/60
    datam['lph'] = datam.apply(lambda row: (row['total_cons']/row['total_time'])*60 if row['total_time']>0 else 'NaN', axis=1)
    datam['avg_speed'] = (datam['total_dist']/datam['total_time'])*0.06
    datam.loc[(datam['Interpolation_status']!='Both_Real')&(datam['total_obs'].isin([0,1])),'max_time_gap'] = (datam['end_time']-datam['start_time']).dt.total_seconds()/60

    return datam

if __name__ == '__main__':
    num_cores = cpu_count()
    final_df_list = p_map(dist_allmods, termid_list, num_cpus=num_cores)
    final_df=pd.concat(final_df_list)
    final_df_dict=final_df.to_dict('records')
    integrated_df_list = p_map(ign_time_int, final_df_dict, num_cpus=num_cores)
    integrated_df=pd.DataFrame(integrated_df_list)
    integrated_df1 = final_data_f(integrated_df)

    integrated_df1.to_csv(output_data_path)
    print('Data saved successfully to this below path:\n{}'.format(output_data_path))


