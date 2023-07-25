import math
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from haversine import haversine, Unit
from tqdm import tqdm
import pytz
import pyreadr
from p_tqdm import p_map
import multiprocessing
import time
import os
tqdm.pandas()
from argparse import ArgumentParser
import argparse
from modules.algorithms import distance_algo,hour_algo,fuel_algo
from modules.data import data_prep_distance,data_prep_hour,data_prep_fuel
from modules.config import termid_class_map   #,allmods_file_input,cst_file_input,output_path  #,termids #,nontermid_path     #distance_output,hour_output,fuel_output



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process data from a file.')
    parser.add_argument('-IA','--input_allmods', type=str, help='Path to the allmods data file.')
    parser.add_argument('-IC','--input_cst', type=str, help='Path to the cst data file.')
    parser.add_argument('-t','--list', nargs=argparse.REMAINDER, type=int, default=[], help='Input list separated by spaces.')
    parser.add_argument('-O','--output', type=str, help='Path to the output data files.')  
    

    args = parser.parse_args()

    df = pyreadr.read_r(args.input_cst)[None]
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
        prev_lat = df.at[row.name - 1, 'lt']
        prev_lon = df.at[row.name - 1, 'lg']
        curr_lat = row['lt']
        curr_lon = row['lg']
        distance = haversine((prev_lat, prev_lon), (curr_lat, curr_lon), unit=Unit.METERS)
        return distance

    df = df.reset_index(drop=True)
    df['Haversine_dist'] = df[['lt','lg']].progress_apply(calculate_distance, axis=1)
    df = df.reset_index(drop=True)   
    df.loc[0,'Haversine_dist'] = 0
    termid_class_map = termid_class_map

    non_termid=[]
    final_dist = pd.DataFrame(); final_hour=pd.DataFrame(); final_fuel=pd.DataFrame()

    df = df[['lt','lg','ts','termid','currentFuelVolumeTank1','currentIgn','regNumb','disthav','Haversine_dist']]
    mods_df = mods_df[['strt','end','lev1', 'lev2', 'fuel','termid','timediff','veh']]


    for i in tqdm(termid_list):
        try:
            df_1 = df[df['termid']==i]
            mods_df_1 = mods_df[mods_df['termid']==i]
            dist_df = distance_algo(df_1,mods_df_1,i)
            hour_df = hour_algo(df_1,mods_df_1)
            fuel_df = fuel_algo(df_1,mods_df_1)
            # try:
            # d_df,h_df,f_df = main(df,mods_df,item)
            final_dist = final_dist.append(dist_df)
            final_hour = final_hour.append(hour_df)
            final_fuel = final_fuel.append(fuel_df)
        except:
            non_termid.append(i)

    final_dist.to_csv(args.output+'\Dist_data.csv')
    final_hour.to_csv(args.output+'\Hour_data.csv')
    final_fuel.to_csv(args.output+'\Fuel_data.csv')
    with open(args.output+'\\nontermid.txt', 'w') as file:
        for i in non_termid:
            file.write(str(i) + '\n')
    print(f'Files saved successfully to this path: {args.output}')


