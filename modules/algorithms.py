import math
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from data import data_prep_distance,data_prep_hour,data_prep_fuel
from config import termid_class_map


def round_to_nearest(number, nearest):
    if nearest in (10, 100, 1000):
        return math.ceil(number / nearest) * nearest
    return number

def internal_agg(df: pd.DataFrame, mods_df: pd.DataFrame, dist, n, termid):

    # df,mods_df = data_prep_distance(input_df,input_mods_df)
    # df = df.reset_index(drop=True)
    list_=[]
    max_ = round_to_nearest(df['cumsum_dist_Sir'].max(),n)+dist
    for i in range(dist,max_,dist):
        temp_dict = {}
        sample = df[i - dist <= df['cumsum_dist_Sir'] <= i]
        if len(sample)!=0:
            total_time = sample['time_diff'].sum();ig_time = sample.query("currentIgn==1")['time_diff'].sum()
            keys=['termid','regNumb','Class','total_samples','start_time','end_time','Initial_level','End_level','Refuel_sum','total_dist','ig_perc']
            values = [sample['termid'].tolist()[0],sample['regNumb'].tolist()[0],termid_class_map[str(termid)],len(sample),
            sample.head(1)['ts'].item(),sample.tail(1)['ts'].item(),sample.head(1)['fuel'].item(),sample.tail(1)['fuel'].item(),sample['REfuel_unique'].sum(),
            sample['disthav'].sum(),ig_time/total_time]
            temp_dict.update(zip(keys,values))
            list_.append(temp_dict)
        else:
            pass

    return list_

def distance_algo(input_df: pd.DataFrame, input_mods_df: pd.DataFrame, termid) -> pd.DataFrame:

    df,mods_df = data_prep_distance(input_df,input_mods_df)   #df_1 , mods_df_1
    if termid_class_map[str(termid)] == 'high_movement':
        list_1 = internal_agg(df,mods_df,1000,1000,termid)
        dist_df = pd.DataFrame(list_1)
        dist_df['Fuel_cons'] = dist_df['Initial_level'] - dist_df['End_level']
        dist_df['Newf_Fuel_cons'] = dist_df['Fuel_cons']+dist_df['Refuel_sum']
        dist_df['Total_time'] = (dist_df['end_time']-dist_df['start_time']).dt.total_seconds()/60
        dist_df['lph'] = (dist_df['Fuel_cons']/dist_df['Total_time'])*60
        dist_df['Median_lph'] = dist_df['lph'].median()
        dist_df['lp_100'] = (dist_df['Fuel_cons']/dist_df['total_dist'])*100000
        dist_df['Median_lp100'] = dist_df['lp_100'].median()
    else:
        dist_df = pd.DataFrame()


    return dist_df    #df,mods_df,

def hour_algo(input_df: pd.DataFrame, input_mods_df: pd.DataFrame):

    df,mods_df = data_prep_hour(input_df,input_mods_df)
    list_=[]
    max_ = (lambda x: round(x+120) if x%120!=0 else x)(df['Cum_Timediff'].max())
    for i in range(120,max_,120):
        temp_dict={}
        sample = df[(df['Cum_Timediff']>(i-120))&(df['Cum_Timediff']<i)]
        if len(sample)!=0:
            total_time = sample['time_diff'].sum();ig_time = sample.query("currentIgn==1")['time_diff'].sum()
            keys = ['term_id','regNumb','total_sample','start_time','end_time','total_time(approx)','start_level','end_level',
                    'Refuel_sum','total_dist','ignPerc','RollmeanLph','RollmeanLp100']
            values=[sample['termid'].iloc[0],sample['regNumb'].iloc[0],len(sample),sample.head(1)['ts'].item(),sample.tail(1)['ts'].item(),
                   sample['time_diff'].sum(),sample.head(1)['fuel'].item(),sample.tail(1)['fuel'].item(),
                    sample['REfuel_unique'].sum(),sample['disthav'].sum(),(ig_time/total_time),sample['lph'].mean(),sample['lp100km'].mean()]
            temp_dict.update(zip(keys,values))
            list_.append(temp_dict)
        else:
            pass
    hour_df = pd.DataFrame(list_)
    hour_df['Fuel_cons'] = hour_df['start_level'] - hour_df['end_level']
    hour_df['New_Fuel_cons'] = hour_df['Fuel_cons']+hour_df['Refuel_sum']
    hour_df['Total_time'] = (hour_df['end_time']-hour_df['start_time']).dt.total_seconds()/60
    hour_df['lph'] = (hour_df['Fuel_cons']/hour_df['Total_time'])*60
    hour_df['Median_lph'] = hour_df['lph'].median()
    hour_df['lp_100'] = (hour_df['Fuel_cons']/hour_df['total_dist'])*100000
    hour_df['Median_lp100'] = hour_df['lp_100'].median()

    return hour_df

def fuel_algo(input_df,input_mods_df):

    df,mods_df = data_prep_fuel(input_df,input_mods_df)
    list_=[]
    max_ = int(df['Bucket'].max())+1
    for i in range(0,max_):
        temp_dict={}
        sample = df[df['Bucket']==i]
        if len(sample)!=0:
            temp_dict['termid'] = sample['termid'].tolist()[0]
            temp_dict['regNumb'] = sample['regNumb'].tolist()[0]
            temp_dict['Total_samples'] = len(sample)
            temp_dict['start_time'] = sample.head(1)['ts'].item()
            temp_dict['end_time'] = sample.tail(1)['ts'].item()
            temp_dict['Initial_level'] = sample.head(1)['currentFuelVolumeTank1'].item()
            temp_dict['End_level'] = sample.tail(1)['currentFuelVolumeTank1'].item()
            temp_dict['Total_Cons'] = sample['Fuel_difference'].sum()
            temp_dict['Total_dist'] = sample['Haversine_dist'].sum()
    #             temp_dict['Total_time'] = sample['time_diff'].sum()
            total_time = sample['time_diff'].sum();ig_time = sample.query("currentIgn==1")['time_diff'].sum()
            temp_dict['ig_perc'] = (ig_time/total_time)
    #             temp_dict['rollMean_lph'] = sample['lph'].mean()
    #             temp_dict['rollMean_lp100km'] = sample['lp100km'].mean()
            list_.append(temp_dict)
        else:
            pass
    fuel_df = pd.DataFrame(list_)
    fuel_df['Total_time'] = (fuel_df['end_time']-fuel_df['start_time']).dt.total_seconds()/60
    fuel_df['lph'] = (fuel_df['Total_Cons']/fuel_df['Total_time'])*60
    fuel_df['Median_lph'] = fuel_df['lph'].median()
    fuel_df['lp_100'] = (fuel_df['Total_Cons']/fuel_df['Total_dist'])*100000
    fuel_df['Median_lp100'] = fuel_df['lp_100'].median()

    return fuel_df
