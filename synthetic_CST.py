import sys
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta, time
from haversine import haversine,haversine_vector, Unit
# import pyarrow.feather as feather
from p_tqdm import p_map
from tqdm import tqdm
from time import sleep
import pytz
import pyreadr
from pathlib import Path
import time
from time import time as t
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

def From_Togrouping(indicator,from_,to_):
    buckets = []
    start_position = None
    for i, value in enumerate(indicator):
        if value == from_:
            if start_position is not None:
                end_position = i
                buckets.append((start_position, end_position))
            start_position = i
        elif value == to_:
            if start_position is not None:
                end_position = i
                buckets.append((start_position, end_position))
                start_position = None
    if start_position is not None:
        buckets.append((start_position, len(indicator)))
    return buckets

def new_fuel(s_time,e_time,s_level,e_level,date):
    total_time = (pd.to_datetime(e_time)-pd.to_datetime(s_time)).total_seconds()/60
    step_size=(e_level-s_level)/total_time
    bucket_size = (pd.to_datetime(date) - pd.to_datetime(s_time)).total_seconds()/60
    new_level = s_level+(bucket_size*step_size)
    return new_level

# Integration of ign start, end times to CST times and Interpolations

def melt_conc(i):
    ign_term = ign[ign['termid']==i];cst_term=cst[cst['termid']==i]
    cst_term=cst_term.reset_index(drop=True);ign_term=ign_term.reset_index(drop=True)
    cst_term['Distance'] = calculate_consecutive_haversine_distances(cst_term)
    melt_ign = pd.melt(ign_term,value_vars=['strt','end'],var_name='Indicator',value_name='ts')
    melt_ign['termid']=str(i);melt_ign['regNumb']=ign_term.head(1)['veh'].item()
    melt_ign.sort_values(by='ts',inplace=True)
#     melt_ign.reset_index(drop=True, inplace=True)
    cst_1 = pd.concat([cst_term,melt_ign],axis=0)
    cst_1.sort_values(by=['ts'],inplace=True)
    cst_1.reset_index(drop=True,inplace=True)
    end_indices = cst_1[cst_1['Indicator'] == 'end'].index
    cst_1.loc[end_indices, 'Distance'] = cst_1['Distance'].shift(-1)
    for ind,row in cst_1.iterrows():
        if (row['Indicator']=='end')&(str(row['Distance'])=='nan'):
            temp_df = cst_term[cst_term['ts']<pd.to_datetime(row['ts'])]
            a_df = cst_term[cst_term['ts']>pd.to_datetime(row['ts'])]
            if (len(temp_df)!=0)&(len(a_df)!=0):
                s_time=temp_df.tail(1)['ts'].item()
                e_time=a_df.head(1)['ts'].item()
                s_level=temp_df.tail(1)['Distance'].item()
                e_level=a_df.head(1)['Distance'].item()
                cst_1.loc[ind,'Distance'] = new_fuel(s_time,e_time,s_level,e_level,row['ts'])
            elif len(temp_df)==0:
                cst_1.loc[ind,'Distance'] = a_df.head(1)['Distance'].item()
            else:
                cst_1.loc[ind,'Distance'] = temp_df.tail(1)['Distance'].item()
    groups = From_Togrouping(cst_1['Indicator'].tolist(),'end','strt')
    for j in groups:
        cst_1.loc[j[0]+1:j[1],'Distance'] = 0
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

    return cst_1

# For shift cutting and Interpolations

def shift_custom_function(group):

    cst_term = new_cst[new_cst['termid']==group.head(1)['termid'].item()]
    group['ts']=pd.to_datetime(group['ts']);cst_term['ts']=pd.to_datetime(cst_term['ts'])
    group.sort_values(by=['ts'],inplace=True)
    time_1 = str(group.head(1)['date'].item())+' 06:00:00'
    temp = cst_term[cst_term['ts']<pd.to_datetime(time_1)];tem = cst_term[cst_term['ts']>pd.to_datetime(time_1)]
    if len(temp)==0:
        time_1_level=tem.head(1)['currentFuelVolumeTank1'].item()
        time_1_dist=tem.head(1)['Distance'].item()
    else:
        time_1_level = new_fuel(temp.tail(1)['ts'].item(),tem.head(1)['ts'].item(),
                               temp.tail(1)['currentFuelVolumeTank1'].item(),
                               tem.head(1)['currentFuelVolumeTank1'].item(),pd.to_datetime(time_1))
        time_1_dist = new_fuel(temp.tail(1)['ts'].item(),tem.head(1)['ts'].item(),
                               temp.tail(1)['Distance'].item(),
                               tem.head(1)['Distance'].item(),pd.to_datetime(time_1))
    time_2 = str(group.head(1)['date'].item())+' 14:00:00'
    temp_1 = cst_term[cst_term['ts']<pd.to_datetime(time_2)];tem_1 = cst_term[cst_term['ts']>pd.to_datetime(time_2)]
    if len(tem_1)==0:
        time_2_level=temp_1.tail(1)['currentFuelVolumeTank1'].item()
        time_2_dist=temp_1.tail(1)['Distance'].item()
    elif len(temp_1)==0:
        time_2_level=tem_1.head(1)['currentFuelVolumeTank1'].item()
        time_2_dist = tem_1.head(1)['Distance'].item()
    else:
        time_2_level = new_fuel(temp_1.tail(1)['ts'].item(),tem_1.head(1)['ts'].item(),
                               temp_1.tail(1)['currentFuelVolumeTank1'].item(),
                               tem_1.head(1)['currentFuelVolumeTank1'].item(),pd.to_datetime(time_2))
        time_2_dist = new_fuel(temp_1.tail(1)['ts'].item(),tem_1.head(1)['ts'].item(),
                               temp_1.tail(1)['Distance'].item(),
                               tem_1.head(1)['Distance'].item(),pd.to_datetime(time_2))
    time_3 = str(group.head(1)['date'].item())+' 22:00:00'
    temp_2 = cst_term[cst_term['ts']<pd.to_datetime(time_3)]
    tem_2 = cst_term[cst_term['ts']>pd.to_datetime(time_3)]
    if len(tem_2)==0:
        time_3_level = temp_2.tail(1)['currentFuelVolumeTank1'].item()
        time_3_dist=temp_2.tail(1)['Distance'].item()
    elif len(temp_2)==0:
        time_3_level=tem_2.head(1)['currentFuelVolumeTank1'].item()
        time_3_dist=tem_2.head(1)['Distance'].item()
    else:
        time_3_level = new_fuel(temp_2.tail(1)['ts'].item(),tem_2.head(1)['ts'].item(),
                               temp_2.tail(1)['currentFuelVolumeTank1'].item(),
                               tem_2.head(1)['currentFuelVolumeTank1'].item(),pd.to_datetime(time_3))
        time_3_dist = new_fuel(temp_2.tail(1)['ts'].item(),tem_2.head(1)['ts'].item(),
                               temp_2.tail(1)['Distance'].item(),
                               tem_2.head(1)['Distance'].item(),pd.to_datetime(time_3))
    temp_df = pd.DataFrame({'ts':[time_1,time_2,time_3],'currentFuelVolumeTank1':[time_1_level,time_2_level,time_3_level],
                           'Distance':[time_1_dist,time_2_dist,time_3_dist]})
    temp_df['termid'] = group.head(1)['termid'].item()
    temp_df['regNumb'] = group.head(1)['regNumb'].item()
    temp_df['ts'] = pd.to_datetime(temp_df['ts'])
    temp_df['mine'] = group.head(1)['mine'].item();temp_df['class'] = group.head(1)['class'].item()
    df = pd.concat([group,temp_df],axis=0)
    df.sort_values(by=['ts'],inplace=True)
    return df

def custom_function(group):

    group_1 = group.groupby('date')
    group_result = group_1.apply(shift_custom_function)
    group_result=group_result.reset_index(drop=True)
    group_result.drop(['timestamp'],axis=1,inplace=True)
    return group_result


def c_func(group):
    group=group.reset_index(drop=True)
    strt_end = From_Togrouping(group['Indicator'].tolist(),'strt','end')
    for j in strt_end:
        group.loc[j[0]:j[1]-1,'currentIgn']= 1
    group.loc[group['currentIgn'].isnull()==True,'currentIgn']=0
    return group


## ID Event Breaking

def binary_parameters(i):
    cst_veh = new_cst_2[new_cst_2['termid']==i]
    cst_veh=cst_veh.reset_index(drop=True)
    cst_veh.sort_values(by=['ts','Indicator'],ascending=[True,False],na_position='first',inplace=True)
    cst_veh['Haversine_dist'] = calculate_consecutive_haversine_distances(cst_veh)
    cst_veh['Time_diff'] = cst_veh['ts'].diff().fillna(pd.Timedelta(minutes=0)).dt.total_seconds() / 60
    cst_veh['Cons_Speed'] = cst_veh['Distance']/cst_veh['Time_diff']
    cst_veh['Cons_Speed'] = cst_veh['Cons_Speed'].fillna(0)
    cst_veh['veh_movement_status'] = 1
    cst_veh.loc[cst_veh['Cons_Speed']<50 , 'veh_movement_status'] = 0

    cst_veh['fuel_consumption']=cst_veh['currentFuelVolumeTank1'].diff().fillna(0)
    cst_veh['Cons_lph']=(cst_veh['fuel_consumption']/cst_veh['Time_diff'])*60
    cst_veh['fuel_movement_status'] = 1
    cst_veh.loc[cst_veh['Cons_lph']==0 , 'fuel_movement_status'] = 0
    cst_veh.loc[0,'fuel_movement_status']=0
    cst_veh.rename(columns={'currentIgn':'ign_status'},inplace=True)
    return cst_veh

binary_to_id = {(1, 1, 1): 'id1',(0, 1, 0): 'id2',(1, 0, 1): 'id3',(0, 1, 1): 'id4',(1, 0, 0): 'id5',
(0, 0, 1): 'id6',(1, 1, 0): 'id7',(0, 0, 0): 'id8'}
def map_binary_to_id(row):
    return binary_to_id[tuple(row)]
def id_attachment(df):
    selected_columns = ['ign_status', 'veh_movement_status', 'fuel_movement_status']
    df['ID_status'] = df[selected_columns].apply(map_binary_to_id, axis=1)
    return df

def cons_id_grouping(list_):
    buckets = []
    current_bucket = []
    for index, value in enumerate(list_):
        if index == 0 or value != list_[index - 1]:
            if current_bucket:
                buckets.append(current_bucket)
            current_bucket = []
        current_bucket.append(index)
    if current_bucket:
        buckets.append(current_bucket)
    return buckets

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

def ign_time_cst(a,b): # output -> final ign time for each event
    # a = ignstatus column ;  b = consecutive Time difference column
    buckets = []
    start_index = None
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
            s = s+(b[j[0]]/5)+(b[j[1]+1]/2.5)
        except:
            s=s+(b[j[0]]/5)
        ign_time=ign_time+s
    return ign_time

def ID_event(j):
    try:
        veh_df = cst_1[cst_1['termid']==j]
        veh_df=veh_df.reset_index(drop=True)
        groups = cons_id_grouping(veh_df['ID_status'].tolist())
        groups=[sublist for sublist in groups if not (len(sublist) == 1 and sublist[0] == 0)]
        list_=[]
        for index,i in enumerate(groups):   # Buckets of consecutive IDs
            if (i[0]==0)&(len(i)!=1):
                sample = veh_df.loc[i[0]:i[-1]]
                id_=veh_df.loc[i[-1],'ID_status']
            elif (i[0]!=0)and(veh_df.loc[i[0],'ID_status'] in ['id1','id3','id7']):
                sample=veh_df.loc[i[0]-1:i[-1]]
                id_ = veh_df.loc[i[-1],'ID_status']
            elif (i[0]!=0)and(veh_df.loc[i[0],'ID_status'] =='id5'):
                if (veh_df.loc[i[0],'Indicator']=='strt')&(i[-1]+1<len(veh_df)):
                    inc = groups[index+1]
                    sample = veh_df.loc[i[0]-1:inc[-1]]
                    id_ = veh_df.loc[inc[-1],'ID_status']
                else:
                    sample = veh_df.loc[i[0]-1:i[-1]]
                    id_ = veh_df.loc[i[-1],'ID_status']
            elif (i[0]!=0)and(veh_df.loc[i[0],'ID_status'] in ['id2','id4','id6','id8'])&(i[-1]+1<=len(veh_df)-1):
                if veh_df.loc[i[-1]+1,'ID_status'] in ['id1','id3','id5','id7']:
                    sample=veh_df.loc[i[0]-1:i[-1]]
                else:
                    sample=veh_df.loc[i[0]-1:i[-1]]
                id_=veh_df.loc[i[-1],'ID_status']
            sample = sample.reset_index(drop=True)
            sample['ts'] = pd.to_datetime(sample['ts'])
            start_time=sample.head(1)['ts'].item()
            end_time=sample.tail(1)['ts'].item()
            sample_list = row_split(str(start_time),str(end_time))    
            l=[]
            for k in range(len(sample_list)):
                temp_dict={}
                sample2=sample[(sample['ts']>=pd.to_datetime(sample_list[k][0]))&(sample['ts']<=pd.to_datetime(sample_list[k][1]))]
                sample2.reset_index(drop=True,inplace=True)
                sample2.loc[0,'Distance']=0
                sample2['new_time_diff'] = sample2['ts'].diff().fillna(pd.Timedelta(minutes=0)).dt.total_seconds() / 60
                ign_cst = ign_time_cst(sample2['ign_status'].tolist(),sample2['new_time_diff'].tolist())
                keys2=['termid','reg_numb','start_time','end_time','total_obs','max_time_gap','initial_level','end_level',
                    'ign_time_cst','total_dist','ID_status','indicator']
                values2=[j,sample2.head(1)['regNumb'].item(),sample_list[k][0],sample_list[k][1],
                        len(sample2),
                        sample2['new_time_diff'].max(),sample2.head(1)['currentFuelVolumeTank1'].item(),sample2.tail(1)['currentFuelVolumeTank1'].item(),
                        ign_cst,sample2['Distance'].sum(),id_,sample.head(1)['Indicator'].item()]             
                temp_dict.update(zip(keys2,values2))
                l.append(temp_dict)
            within_df = pd.DataFrame(l)
            within_df = within_df.reset_index(drop=True)
            list_.append(within_df)
        ff=pd.concat(list_)
        ff['start_time'] = pd.to_datetime(ff['start_time'])
        ff['end_time']=pd.to_datetime(ff['end_time'])
        ff.drop_duplicates(subset=['end_time'],keep='first',inplace=True)
        
        # Ignition time Calculation from Ignition Master Data
        ff.reset_index(drop=True,inplace=True)
        veh_ign = ign[ign['termid']==j]
        veh_ign = veh_ign.reset_index(drop=True)
        veh_f_df_dict = ff.to_dict('records')
        for row in veh_f_df_dict:
            ign_ = veh_ign.loc[(((veh_ign['strt']<=row['end_time'])&(veh_ign['strt']>=row['start_time'])) | ((veh_ign['end']<=row['end_time'])&(veh_ign['end']>=row['start_time'])) | ((veh_ign['strt']<=row['start_time'])&(veh_ign['end']>=row['end_time'])))]
            ign_.loc[ign_['strt']<row['start_time'],'strt']=row['start_time']
            ign_.loc[ign_['end']>row['end_time'],'end']=row['end_time']
            ign_['dur(mins)']=(ign_['end']-ign_['strt'])/timedelta(minutes=1)
            row['ign_time_igndata'] = sum(ign_['dur(mins)'])
        ff1 = pd.DataFrame(veh_f_df_dict)
        return ff1
    except Exception as e:
        print(f"An error Occured: {e}")
        sys.exit(1)

def final_cal(df):
    df=df.reset_index(drop=True)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time']=pd.to_datetime(df['end_time'])
    df['total_cons']=df['initial_level']-df['end_level']
    df['lp100k'] = df.apply(lambda row: (row['total_cons']/row['total_dist'])*100000 if row['total_dist'] > 0 else 'NaN', axis=1)
    df['total_time'] = (df['end_time']-df['start_time']).dt.total_seconds()/60
    df['lph'] = df.apply(lambda row: (row['total_cons']/row['total_time'])*60 if row['total_time']>0 else 'NaN', axis=1)
    df['avg_speed'] = (df['total_dist']/df['total_time'])*0.06
    df.loc[(df['ID_status']=='id2')|(df['ID_status']=='id4')|(df['ID_status']=='id6')|(df['ID_status']=='id8'),'ign_time_igndata']=0
    df.loc[(df['ID_status']=='id5')&(df['total_cons']!=0) , 'ID_status']='id3'
    df.loc[(df['ID_status']=='id7')&(df['total_cons']!=0) , 'ID_status']='id1'
    df.loc[(df['ID_status']=='id8')&(df['total_cons']!=0) , 'ID_status']='id6'
    df.loc[(df['ID_status']=='id6')&(df['total_cons']==0) , 'ID_status']='id8'
    df.loc[(df['ID_status']=='id3')&(df['total_cons']==0) , 'ID_status']='id5'
    df.loc[(df['ID_status']=='id1')&(df['total_cons']==0) , 'ID_status']='id7'
    df.loc[(df['ID_status']=='id4')&(df['total_cons']==0) , 'ID_status']='id2'
    df.loc[(df['ID_status'].isin(['id1', 'id3', 'id5', 'id7'])) & (df['ign_time_igndata'] == 0), 'ign_time_igndata'] = df['ign_time_cst']
    return df


if __name__ == '__main__':

    # num_cores = cpu_count()
    if len(sys.argv) < 3:
      print('InputFilesError: You need to provide the path of the RDS files as input.\nCST data followed by ignition data.')
    else:
      infile_cst, infile_igtn = Path(sys.argv[1]), Path(sys.argv[2])

      # Check validity of both args at once
      if infile_cst.suffix == infile_igtn.suffix != '.RDS':
        print('FileFormatError: Only RDS files applicable as input\nExiting....')
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

      new_cst = pd.concat([melt_conc(termid) for termid in tqdm(termid_list)])   # melting ign, concat and Interpolations
      grouped = new_cst.groupby(['termid'])
      new_cst_1 = grouped.progress_apply(custom_function)                         #shift cut and Interpolations
      new_cst_1=new_cst_1.reset_index(drop=True)
      new_cst_1['termid']=new_cst_1['termid'].astype('int32')                   # to make dataframe memory efficient
      new_cst_1['mine']=new_cst_1['mine'].astype('category')
      new_cst_1['class']=new_cst_1['class'].astype('category')
      new_cst_1['Indicator']=new_cst_1['Indicator'].astype('category')
      grouped_1 = new_cst_1.groupby(['termid'])
      new_cst_2 = grouped_1.progress_apply(c_func)                             # Ignition status markings
      new_cst_2=new_cst_2.reset_index(drop=True)

      tr_final_df = pd.concat([binary_parameters(i) for i in tqdm(termid_list)])     # Tri parameters Binary values
      cst_1 = id_attachment(tr_final_df)                                             # ID status add
      cst_1 = cst_1[['regNumb','currentFuelVolumeTank1','ign_status','ts','termid','Distance','Indicator','ID_status']]
      cst_1['termid']=cst_1['termid'].astype('int32')
      ign['termid']=ign['termid'].astype('int32')
      cst_1['Indicator']=cst_1['Indicator'].astype('category')
      cst_1['ID_status']=cst_1['ID_status'].astype('category')
      final_df = pd.concat([ID_event(termid) for termid in tqdm(termid_list[:10])])      # event algorithm
      final_df1 = final_cal(final_df)                                


    # Error Logging for Output Files
      if len(sys.argv) == 3:
        new_cst_2.to_csv('New_Synthetic_CST.csv')
        final_df1.to_csv('ID_event_data.csv')
        print('Data saved successfully to the above path')
      elif len(sys.argv)==4:
          print('FileArgumentsError: Kindly put 4 file arguments. There are 3.\nExiting....')
          sys.exit(0)
      elif len(sys.argv) == 5:
        outfile1 = Path(sys.argv[3])
        print(str(outfile1).split('\\')[-1])
        outfile2 = Path(sys.argv[4])
        if (outfile1.suffix != '.csv')or(outfile2.suffix != '.csv'):
          print('OutputFilesFormatError: Need to write outputs to CSV files only\nExiting....')
          sys.exit(0)
        elif (outfile1 == outfile2)or(str(outfile1).split('\\')[-1]==str(outfile2).split('\\')[-1]):
          print("OutputFilesNameError: Output file Paths or Names can't be same\nExiting...")
          sys.exit(0)
          
        new_cst_2.to_csv(outfile1)
        final_df1.to_csv(outfile2)
        print(f'Synthetic CST and Event data are successfully saved to \n{outfile1} and {outfile2}.')

      # Check for extra args
      else:
        print('Supports atleast 2 or 4 file arguments.')
