import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta, time
from haversine import haversine, Unit
from haversine import haversine_vector, Unit
from tqdm import tqdm
from time import sleep
from pathlib import Path
import pytz
import pyreadr
import time
from time import time as t
from itertools import combinations, permutations
import math
import os
tqdm.pandas()
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 150)
from multiprocess import cpu_count


def calculate_consecutive_haversine_distances(df):
    distances = []
    for i in range(1, len(df)):
        lat1, lon1 = df.at[i-1, 'lt'], df.at[i-1, 'lg']
        lat2, lon2 = df.at[i, 'lt'], df.at[i, 'lg']
        distance = haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)
        distances.append(distance)
    distances.insert(0,0)
    return distances

def disp_cst(i):
    term_df = cst[cst['regNumb']==i]
    term_df['Time_diff'] = term_df['ts'].diff().fillna(pd.Timedelta(minutes=0)).dt.total_seconds() / 60
    term_df.reset_index(drop=True,inplace=True)
    term_df['Distance'] = calculate_consecutive_haversine_distances(term_df)
    term_df['cum_distance'] = term_df['Distance'].cumsum().fillna(0)
    disp_df = disp[disp['regNumb']==i]
    if len(disp_df)!=0:
#         disp_df['Quantity'] = disp_df['Quantity'].str.replace(',','').astype(float)
        con = pd.concat([term_df,disp_df],axis=0)
        con.sort_values(by=['ts'],inplace=True)
        con.loc[con['termid'].isnull(),'termid']=term_df.head(1)['termid'].item()
        con['mine'] = term_df.head(1)['mine'].item()
        # con['Station Name'] = term_df.head(1)['Station Name'].item()
        con['class'] = term_df.head(1)['class'].item()
        return con
    else:
        return term_df

def new_fuel(s_time,e_time,s_level,e_level,date):
    total_time = (pd.to_datetime(e_time)-pd.to_datetime(s_time)).total_seconds()/60
    step_size=(e_level-s_level)/total_time
    bucket_size = (pd.to_datetime(date) - pd.to_datetime(s_time)).total_seconds()/60
    new_level = s_level+(bucket_size*step_size)
    return new_level

def refuel_end_injection(i):
    term_df = disp_cst[disp_cst['termid']==i]
    term_df.reset_index(drop=True,inplace=True)
    if (term_df.loc[0,'Refuel_status']=='Refuel')&(term_df.loc[term_df.index[-1],'Refuel_status']=='Refuel'):
        term_df.drop([0,term_df.index[-1]],axis=0,inplace=True)
    elif term_df.loc[0,'Refuel_status']=='Refuel':
        term_df.drop(0,axis=0,inplace=True)
    elif term_df.loc[term_df.index[-1],'Refuel_status']=='Refuel':
        term_df.drop(term_df.index[-1],inplace=True)
    else:
        pass
    term_df.reset_index(drop=True,inplace=True)
#     count = count+len(term_df)
    injected_data=[]
    for ind,j in term_df.iterrows():
        if (j['Refuel_status']=='Refuel'):
            term_df.loc[ind,'currentFuelVolumeTank1'] = term_df.loc[ind-1,'currentFuelVolumeTank1']
            refuel_end_time = term_df.loc[ind,'ts'] + timedelta(minutes=20)
            refuel_end_level = term_df.loc[ind-1,'currentFuelVolumeTank1'] + term_df.loc[ind,'Quantity']
            refuel_start_cum_distance = new_fuel(term_df.loc[ind-1,'ts'],term_df.loc[ind+1,'ts'],
                                                term_df.loc[ind-1,'cum_distance'],term_df.loc[ind+1,'cum_distance'],
                                                 term_df.loc[ind,'ts'])
            term_df.loc[ind,'cum_distance'] = refuel_start_cum_distance
            injected_data.append({'termid':i,'regNumb':term_df.head(1)['regNumb'].item(),'ts':refuel_end_time,
                                  'currentFuelVolumeTank1':refuel_end_level,'Refuel_status':'Refuel_end'})
    injected_df = pd.DataFrame(injected_data)
#     normal_df=normal_df.append(term_df)
    concat_df = pd.concat([term_df,injected_df],axis=0,ignore_index=True)
    concat_df.sort_values(by=['termid','ts'],inplace=True)
    return concat_df

def refuel_end_cum_distance(i):
# for i in tqdm(termid_list):
    term_df = disp_cst1[disp_cst1['termid']==i]
    term_df.reset_index(drop=True,inplace=True)
    for ind,j in term_df.iterrows():
        if j['Refuel_status']=='Refuel_end':
            a = term_df[term_df['ts']<pd.to_datetime(j['ts'])]
            b = term_df[term_df['ts']>pd.to_datetime(j['ts'])]
            end_cum_distance = new_fuel(a.tail(1)['ts'].item(),b.head(1)['ts'].item(),a.tail(1)['cum_distance'].item(),
                                       b.head(1)['cum_distance'].item(),j['ts'])
            term_df.loc[ind,'cum_distance'] = end_cum_distance
    return term_df

def melt_conc(i):
    ign_term = ign[ign['termid']==i];cst_term=disp_cst2[disp_cst2['termid']==i]
    cst_term=cst_term.reset_index(drop=True);ign_term=ign_term.reset_index(drop=True)
#     cst_term['Distance'] = calculate_consecutive_haversine_distances(cst_term)
#     cst_term['cum_distance'] = cst_term['Distance'].cumsum().fillna(0)
    melt_ign = pd.melt(ign_term,value_vars=['strt','end'],var_name='Indicator',value_name='ts')
    melt_ign['termid']=str(i);melt_ign['regNumb']=ign_term.head(1)['veh'].item()
    melt_ign.sort_values(by='ts',inplace=True)
#     melt_ign.reset_index(drop=True, inplace=True)
    cst_1 = pd.concat([cst_term,melt_ign],axis=0)
    cst_1.sort_values(by=['ts'],inplace=True)
    cst_1.reset_index(drop=True,inplace=True)
    end_indices = cst_1[cst_1['Indicator'] == 'end'].index
#     cst_1.loc[end_indices, 'Distance'] = cst_1['Distance'].shift(-1)
    cst_1.loc[end_indices, 'cum_distance'] = cst_1['cum_distance'].shift(-1)
    for ind,row in cst_1.iterrows():
        if (row['Indicator'] in ('end','strt'))&(str(row['cum_distance'])=='nan'):
            temp_df = cst_term[cst_term['ts']<pd.to_datetime(row['ts'])]
            a_df = cst_term[cst_term['ts']>pd.to_datetime(row['ts'])]
            if (len(temp_df)!=0)&(len(a_df)!=0):
                s_time=temp_df.tail(1)['ts'].item()
                e_time=a_df.head(1)['ts'].item()
#                 s_level=temp_df.tail(1)['Distance'].item()
                s_level_1=temp_df.tail(1)['cum_distance'].item()
#                 e_level=a_df.head(1)['Distance'].item()
                e_level_1=a_df.head(1)['cum_distance'].item()
#                 cst_1.loc[ind,'Distance'] = new_fuel(s_time,e_time,s_level,e_level,row['ts'])
                cst_1.loc[ind,'cum_distance'] = new_fuel(s_time,e_time,s_level_1,e_level_1,row['ts'])
            elif len(temp_df)==0:
#                 cst_1.loc[ind,'Distance'] = a_df.head(1)['Distance'].item()
                cst_1.loc[ind,'cum_distance'] = a_df.head(1)['cum_distance'].item()
            else:
#                 cst_1.loc[ind,'Distance'] = temp_df.tail(1)['Distance'].item()
                cst_1.loc[ind,'cum_distance'] = temp_df.tail(1)['cum_distance'].item()
#     groups = Off_On_grouping(cst_1['Indicator'].tolist())
#     for j in groups:
#         cst_1.loc[j[0]+1:j[1],'Distance'] = 0
    cst_1.sort_values(by=['ts','Indicator'],inplace=True)
    cst_1.reset_index(drop=True,inplace=True)
    fil_strt=cst_1.query("Indicator=='strt'")
    for ind,row in fil_strt.iterrows():
#         if ind!=0:
        temp_df = cst_term[cst_term['ts']<pd.to_datetime(row['ts'])]
        a_df = cst_term[cst_term['ts']>pd.to_datetime(row['ts'])]
        if (len(temp_df)!=0)&(len(a_df)!=0):
            s_time=temp_df.tail(1)['ts'].item()
            e_time=cst_term[cst_term['ts']>pd.to_datetime(row['ts'])].head(1)['ts'].item()
            s_level=temp_df.tail(1)['currentFuelVolumeTank1'].item()
            e_level=cst_term[cst_term['ts']>pd.to_datetime(row['ts'])].head(1)['currentFuelVolumeTank1'].item()
            cst_1.loc[ind,'currentFuelVolumeTank1']= new_fuel(s_time,e_time,s_level,e_level,row['ts'])
        elif len(temp_df)==0:
            cst_1.loc[ind,'currentFuelVolumeTank1'] = a_df.head(1)['currentFuelVolumeTank1'].item()
        else:
            cst_1.loc[ind,'currentFuelVolumeTank1'] = temp_df.tail(1)['currentFuelVolumeTank1'].item()
    end_indices = cst_1[cst_1['Indicator'] == 'end'].index
    cst_1.loc[end_indices, 'currentFuelVolumeTank1'] = cst_1['currentFuelVolumeTank1'].shift(-1)
    cst_1['termid'] = cst_1['termid'].astype(int)
    cst_1.sort_values(by=['ts','Indicator'],na_position='first',inplace=True)
    cst_1.reset_index(drop=True,inplace=True)
    duplicated = [int(i) - 1 for i in cst_1[cst_1['ts'].duplicated()].index.tolist()]
    for l in duplicated:
        cst_1.loc[l,'Indicator'] = cst_1.loc[l+1,'Indicator']
    cst_1.drop_duplicates(subset=['ts'],keep='first',inplace=True)
    cst_1['mine']=cst_term.head(1)['mine'].item()
    cst_1['class'] = cst_term.head(1)['mine'].item()

    return cst_1

def shift_custom_function(group):
    cst_term = new_cst[new_cst['termid']==group.head(1)['termid'].item()]
#     print(len(cst_term))
#     print('len of day 1204000258 from new_cst',len(group))
    group['ts']=pd.to_datetime(group['ts']);cst_term['ts']=pd.to_datetime(cst_term['ts'])
    group.sort_values(by=['ts'],inplace=True)
    time_1 = str(group.head(1)['date'].item())+' 06:00:00'
    temp = cst_term[cst_term['ts']<pd.to_datetime(time_1)];tem = cst_term[cst_term['ts']>pd.to_datetime(time_1)]
    if len(temp)==0:
        time_1_level=tem.head(1)['currentFuelVolumeTank1'].item()
#         time_1_dist=tem.head(1)['Distance'].item()
        time_1_cum_dist=tem.head(1)['cum_distance'].item()
    else:
        time_1_level = new_fuel(temp.tail(1)['ts'].item(),tem.head(1)['ts'].item(),
                               temp.tail(1)['currentFuelVolumeTank1'].item(),
                               tem.head(1)['currentFuelVolumeTank1'].item(),pd.to_datetime(time_1))
#         time_1_dist = new_fuel(temp.tail(1)['ts'].item(),tem.head(1)['ts'].item(),
#                                temp.tail(1)['Distance'].item(),
#                                tem.head(1)['Distance'].item(),pd.to_datetime(time_1))
        time_1_cum_dist=new_fuel(temp.tail(1)['ts'].item(),tem.head(1)['ts'].item(),
                               temp.tail(1)['cum_distance'].item(),
                               tem.head(1)['cum_distance'].item(),pd.to_datetime(time_1))
    time_2 = str(group.head(1)['date'].item())+' 14:00:00'
    temp_1 = cst_term[cst_term['ts']<pd.to_datetime(time_2)];tem_1 = cst_term[cst_term['ts']>pd.to_datetime(time_2)]
    if len(tem_1)==0:
        time_2_level=temp_1.tail(1)['currentFuelVolumeTank1'].item()
#         time_2_dist=temp_1.tail(1)['Distance'].item()
        time_2_cum_dist=temp_1.tail(1)['cum_distance'].item()
    elif len(temp_1)==0:
        time_2_level=tem_1.head(1)['currentFuelVolumeTank1'].item()
#         time_2_dist = tem_1.head(1)['Distance'].item()
        time_2_cum_dist=tem_1.head(1)['cum_distance'].item()
    else:
        time_2_level = new_fuel(temp_1.tail(1)['ts'].item(),tem_1.head(1)['ts'].item(),
                               temp_1.tail(1)['currentFuelVolumeTank1'].item(),
                               tem_1.head(1)['currentFuelVolumeTank1'].item(),pd.to_datetime(time_2))
#         time_2_dist = new_fuel(temp_1.tail(1)['ts'].item(),tem_1.head(1)['ts'].item(),
#                                temp_1.tail(1)['Distance'].item(),
#                                tem_1.head(1)['Distance'].item(),pd.to_datetime(time_2))
        time_2_cum_dist = new_fuel(temp_1.tail(1)['ts'].item(),tem_1.head(1)['ts'].item(),
                               temp_1.tail(1)['cum_distance'].item(),
                               tem_1.head(1)['cum_distance'].item(),pd.to_datetime(time_2))
#     else:
#         time_2_level = 0;time_2_dist=0
    time_3 = str(group.head(1)['date'].item())+' 22:00:00'
    temp_2 = cst_term[cst_term['ts']<pd.to_datetime(time_3)]
    tem_2 = cst_term[cst_term['ts']>pd.to_datetime(time_3)]
    if len(tem_2)==0:
        time_3_level = temp_2.tail(1)['currentFuelVolumeTank1'].item()
#         time_3_dist=temp_2.tail(1)['Distance'].item()
        time_3_cum_dist=temp_2.tail(1)['cum_distance'].item()
    elif len(temp_2)==0:
#         print(group.head(1)['termid'].item(),group.head(1)['date'].item())
        time_3_level=tem_2.head(1)['currentFuelVolumeTank1'].item()
#         time_3_dist=tem_2.head(1)['Distance'].item()
        time_3_cum_dist=tem_2.head(1)['cum_distance'].item()
    else:
        time_3_level = new_fuel(temp_2.tail(1)['ts'].item(),tem_2.head(1)['ts'].item(),
                               temp_2.tail(1)['currentFuelVolumeTank1'].item(),
                               tem_2.head(1)['currentFuelVolumeTank1'].item(),pd.to_datetime(time_3))
#         time_3_dist = new_fuel(temp_2.tail(1)['ts'].item(),tem_2.head(1)['ts'].item(),
#                                temp_2.tail(1)['Distance'].item(),
#                                tem_2.head(1)['Distance'].item(),pd.to_datetime(time_3))
        time_3_cum_dist = new_fuel(temp_2.tail(1)['ts'].item(),tem_2.head(1)['ts'].item(),
                               temp_2.tail(1)['cum_distance'].item(),
                               tem_2.head(1)['cum_distance'].item(),pd.to_datetime(time_3))
    temp_df = pd.DataFrame({'ts':[time_1,time_2,time_3],'currentFuelVolumeTank1':[time_1_level,time_2_level,time_3_level],
                           'cum_distance':[time_1_cum_dist,time_2_cum_dist,time_3_cum_dist]})
    temp_df['termid'] = group.head(1)['termid'].item()
    temp_df['regNumb'] = group.head(1)['regNumb'].item()
    temp_df['ts'] = pd.to_datetime(temp_df['ts'])
    temp_df['mine'] = group.head(1)['mine'].item();temp_df['class'] = group.head(1)['class'].item()
    df = pd.concat([group,temp_df],axis=0)
    df.sort_values(by=['ts','direction'],na_position='last',inplace=True)
    df.drop_duplicates(subset=['ts'],inplace=True)
    return df

def custom_function(group):

    group_1 = group.groupby('date')
    group_result = group_1.apply(shift_custom_function)
#     print('group result len: ',group_result.shape)
    group_result=group_result.reset_index(drop=True)
#     print('group result len: ',len(group_result))
    group_result.drop(['timestamp'],axis=1,inplace=True)
    group_result['mine']= group.head(1)['mine'].item()
    group_result['class'] = group.head(1)['mine'].item()
    group_result['Distance'] = group_result['cum_distance'].diff().fillna(0)
    return group_result


if __name__ == '__main__':

    # print(len(sys.argv))
    # print(sys.argv[0],sys.argv[1])
    num_cores = cpu_count()
    if len(sys.argv) < 4:
      print('InputFilesError: You need to provide the path of RDS cst and ign files and Hectronic csv as input.\nCST data followed by ignition data followed by Hectronics Dispense Data.\nExiting....')
      sys.exit(0)
    else:
      infile_cst, infile_igtn,disp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

      # Check validity of both args at once
      if (infile_cst.suffix == infile_igtn.suffix != '.RDS') or (disp.suffix!='.csv'):
        print('FileFormatError: Only RDS files for cst/ign and csv for Hec data applicable as input\nExiting....')
        sys.exit(0)

      cst = pyreadr.read_r(infile_cst)[None]
      ign = pyreadr.read_r(infile_igtn)[None]
      cst['ts'] = cst['ts'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
      cst['date'] = pd.to_datetime(cst['ts']).dt.date.astype(str)
      cst.rename(columns={'latitude':'lt', 'longitude':'lg'}, inplace=True)
      faulty_fuel = cst[cst['currentFuelVolumeTank1'].isnull()]['regNumb'].unique().tolist()
      cst = cst[~cst['regNumb'].isin(faulty_fuel)]
      ign.rename(columns={'stop':'end'}, inplace=True)
      ign['strt'] = ign['strt'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
      ign['end'] = ign['end'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
      ign['termid'] = ign['termid'].astype(int)
      ign = ign[['termid','veh','strt','end']]
      termid_list = cst[cst['termid'].isin(ign['termid'])]['termid'].unique().tolist()
      disp = pd.read_csv(disp)
      disp.rename(columns={'Vehicle Number':'regNumb','Date':'date','Time Stamp':'ts'},inplace=True)
      disp=disp[disp['regNumb'].isin(cst['regNumb'])][['ts','date','Station Name','regNumb','Quantity']]
      disp['Refuel_status'] = 'Refuel'
      disp['ts'] = pd.to_datetime(disp['ts'], format='%m/%d/%Y, %I:%M:%S %p')
      disp['Quantity'] = disp['Quantity'].str.replace(',','').astype(float)
      disp = disp[disp['Quantity']>20]
      regNumb_list = cst[cst['termid'].isin(ign['termid'])]['regNumb'].unique().tolist()

      if len(cst)==0:
        print('CstDataError: Fuel levels API values error/Blank. Kindly pass a valid cst data.\nExiting....')
        sys.exit(0)

      disp_cst = pd.concat([disp_cst(i) for i in tqdm(regNumb_list[:1])])
      disp_cst1 = pd.concat([refuel_end_injection(i) for i in tqdm(termid_list[:1])])
      disp_cst2 = pd.concat([refuel_end_cum_distance(i) for i in tqdm(termid_list[:1])])
      new_cst = pd.concat([melt_conc(termid) for termid in tqdm(termid_list[:1])])
      new_cst['termid']=new_cst['termid'].astype(int)
      new_cst['date'] = new_cst['ts'].dt.date
      grouped = new_cst.groupby('termid')
      new_cst_1=grouped.progress_apply(custom_function)
      new_cst_1=new_cst_1.reset_index(drop=True)
      new_cst_1['date'] = new_cst_1['ts'].dt.date
      new_cst_1.drop(['Time_diff','Station Name'],axis=1,inplace=True)
    #   print(len(new_cst_1))


    # Error Logging for Output Files
      if len(sys.argv) == 4:
        new_cst_1.to_csv('New_Synthetic_CST.csv')
        # final_df1.to_csv('ID_event_data.csv')
        print('Data saved successfully into your Working Directory.')
    #   elif len(sys.argv)==5:
    #       print('FileArgumentsError: Kindly put 4 file arguments. There are 3.\nExiting....')
    #       sys.exit(0)
      elif len(sys.argv) == 5:
        outfile1 = Path(sys.argv[4])
        # print(str(outfile1).split('\\')[-1])
        # outfile2 = Path(sys.argv[4])
        # if (outfile1.suffix != '.csv')or(outfile2.suffix != '.csv'):
        #   print('OutputFilesFormatError: Need to write outputs to CSV files only\nExiting....')
        #   sys.exit(0)
        # elif (outfile1 == outfile2)or(str(outfile1).split('\\')[-1]==str(outfile2).split('\\')[-1]):
        #   print("OutputFilesNameError: Output file Paths or Names can't be same\nExiting...")
        #   sys.exit(0)

        new_cst_1.to_csv(outfile1)
        # final_df1.to_csv(outfile2)
        print(f' Enriched CST is successfully saved to below path: \n{outfile1}.')
      # Check for extra args
      else:
        print('Supports atleast 3 or 4 file arguments.')
