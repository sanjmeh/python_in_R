from modules.config import termid_class_map   #,allmods_file_input,cst_file_input,output_path  #,termids #,nontermid_path,distance_output,hour_output,fuel_output
from modules.algorithms import distance_algo,hour_algo,fuel_algo
from datetime import datetime
from haversine import haversine, Unit
from tqdm import tqdm
import pandas as pd 
import argparse
import pytz
import pyreadr
import time

import warnings
warnings.filterwarnings("ignore")
tqdm.pandas()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process data from a file.')
    parser.add_argument('-IA','--input_allmods', type=str, help='Path to the allmods data file.')
    parser.add_argument('-IC','--input_cst', type=str, help='Path to the cst data file.')
    parser.add_argument('-O','--output', type=str, help='Path to the output data folder.')
    parser.add_argument('-t','--list', nargs=argparse.REMAINDER, type=int, default=[], help='Input list separated by spaces.')
    

    args = parser.parse_args()
    
    
    def timestamp_(date):
        formatted_datetime = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
        #print('T1')
        return formatted_datetime

    try:
        df_chunks = []
        for chunk in pd.read_csv(args.input_cst, chunksize=10000):
            df_chunks.append(chunk)       
        df = pd.concat(df_chunks, ignore_index=True)   
        df['ts'] = df['ts'].progress_apply(timestamp_)    

    except:
        df = pyreadr.read_r(args.input_cst)[None]       

    try:
        mod_chunk=[]
        for ch in pd.read_csv(args.input_allmods,chunksize=10000):
            mod_chunk.append(ch)
        mods_df = pd.concat(mod_chunk, ignore_index=True)
        mods_df['strt'] = mods_df['strt'].progress_apply(timestamp_)
        mods_df['end'] = mods_df['end'].progress_apply(timestamp_)
    except:
        mods_df = pyreadr.read_r(args.input_allmods)[None]
    
    if len(args.list)!=0:
        df = df[df['termid'].isin(args.list)]
        mods_df = mods_df[mods_df['termid'].isin(args.list)]
        termid_list = args.list
    else:
        termid_list = df['termid'].unique().tolist()
        pass

    def calculate_distance(row):
        if row.name == 0:
            return 0
        prev_lat = df_1.at[row.name - 1, 'lt']
        prev_lon = df_1.at[row.name - 1, 'lg']
        curr_lat = row['lt']
        curr_lon = row['lg']
        distance = haversine((prev_lat, prev_lon), (curr_lat, curr_lon), unit=Unit.METERS)
        return distance
    
    def Utc_to_Ist(utc_time1):

        utc_time = datetime.strptime(str(utc_time1), "%Y-%m-%d %H:%M:%S")
        ist_timezone = pytz.timezone("Asia/Kolkata")
        ist_time = utc_time.replace(tzinfo=pytz.UTC).astimezone(ist_timezone).strftime("%Y-%m-%d %H:%M:%S")
        return ist_time

    termid_class_map = termid_class_map

    non_termid=[]
    final_dist = pd.DataFrame(); final_hour=pd.DataFrame(); final_fuel=pd.DataFrame()  #;before_bucket=pd.DataFrame();before_bucket_m=pd.DataFrame()

    df = df[['lt','lg','ts','termid','currentFuelVolumeTank1','currentIgn','regNumb','disthav']]
    mods_df = mods_df[['strt','end','lev1', 'lev2', 'fuel','termid','timediff','veh']]
    df['ts'] = df['ts'].progress_apply(Utc_to_Ist)

    mods_df[['strt','end']] = mods_df[['strt','end']].applymap(Utc_to_Ist)

    start_time = time.time()


    # just_filtered_mod = pd.DataFrame()
    for i in tqdm(termid_list):
        try:
            df_1 = df[df['termid']==i]
            df_1 = df_1.reset_index(drop=True)
            df_1['Haversine_dist'] = df_1[['lt','lg']].apply(calculate_distance, axis=1)
            df_1 = df_1.reset_index(drop=True)   
            df_1.loc[0,'Haversine_dist'] = 0
            mods_df_1 = mods_df[mods_df['termid']==i]
            dist_df = distance_algo(df_1,mods_df_1,i)   # before_bucket_1,before_bucket_m1,
            hour_df = hour_algo(df_1,mods_df_1)
            fuel_df = fuel_algo(df_1,mods_df_1)
            # try:
            # d_df,h_df,f_df = main(df,mods_df,item)
            # just_filtered_mod = just_filtered_mod.append(mods_df_1)
            # before_bucket = before_bucket.append(before_bucket_1)
            # before_bucket_m = before_bucket_m.append(before_bucket_m1)
            final_dist = final_dist.append(dist_df)
            final_hour = final_hour.append(hour_df)
            final_fuel = final_fuel.append(fuel_df)
        except:
            non_termid.append(i)
    end_time = time.time()
    execution_time = end_time - start_time
    
    final_dist.to_csv(args.output+'/Dist_data_dump.csv')
    # before_bucket.to_csv(args.output+'/Before_Bucket.csv')
    # before_bucket_m.to_csv(args.output+'/Before_Bucket_Mod.csv')
    final_hour.to_csv(args.output+'/Hour_data_dump.csv')
    final_fuel.to_csv(args.output+'/Fuel_data_dump.csv')
    with open(args.output+'/nontermid_dump.txt', 'w') as file:
        for i in non_termid:
            file.write(str(i) + '\n')
    print(f'Files saved successfully to this path: {args.output}')
    print(f'Total execution time:{round(execution_time,2)} seconds')


