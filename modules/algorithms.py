from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import linregress

from matplotlib.dates import date2num

from .config import BASE_DIR, VEHICLE_DIR

def detect_dg_idling(df_event, df_elm, df_fuel, dl):
    output_df = {
        'start_ts': [],
        'end_ts': [],
        'duration_idled': [],
        'fuel_consumed': []
    }

    for i, row in tqdm(df_event.iterrows(), total=len(df_event)):
        if row.type == 5:
            ts_1 = row.ts
            ts_2 = None
            for j in range(i, len(df_event)):
                if df_event.loc[j, 'type'] == 6:
                    ts_2 = df_event.loc[j, 'ts']
                    break
            ign_df = df_elm[(df_elm['ts']>=ts_1) & (df_elm['ts']<=ts_2)]
            idl_df = ign_df[ign_df['VR']<1.0]

            list_of_df = [d for _, d in idl_df.groupby(idl_df.index - np.arange(len(idl_df)))]

            for idl in list_of_df:
                try:
                    idl = idl.reset_index(drop=True)
                except Exception as e:
                    pass
                
                i_start = idl.loc[0, 'ts']
                i_end = idl.loc[len(idl)-1, 'ts']
                if len(idl)>1 and (i_end - i_start) > dl:
                    fuel_cdf = df_fuel[(df_fuel['ts']>=i_start) & (df_fuel['ts']<=i_end)]
                    fuel_val = fuel_cdf.sort_values(by='ts').rV.values
                    fuel_consumed = fuel_val[0] - fuel_val[-1]

                    output_df['start_ts'].append(i_start)
                    output_df['end_ts'].append(i_end)
                    output_df['duration_idled'].append(str(i_end-i_start))
                    output_df['fuel_consumed'].append(fuel_consumed)

                    # print(f"\n DG Idling from: {i_start} to: {i_end} for duration: {str(i_end-i_start)}; fuel consumed: {fuel_consumed}L")
                    # print(idl)
    
    print("Saving output...")
    pd.DataFrame(output_df).to_csv(BASE_DIR+'idling_output.csv', index=False)
    print("Output saved.")

def detect_fuel_drain(df_event, df_fuel):
    output_df = {
        'start_ts': [],
        'end_ts': [],
        'duration_drained': [],
        'fuel_drained': []
    }

    n = 2
    de = pd.Timedelta('2 minutes')
    date_dict = {
        'index': [],
        'ts': []
    }

    df_fuel.rV = df_fuel.rV.rolling(3, min_periods=1).mean()

    ignition_list = []

    for i, row in df_event.iterrows():
        if row.type == 5:
            ts_1 = row.ts
            ts_2 = None
            for j in range(i, len(df_event)):
                if df_event.loc[j, 'type'] == 6:
                    ts_2 = df_event.loc[j, 'ts']
                    break
            ignition_list.append((ts_1, ts_2))

    for i, row in tqdm(df_fuel.iterrows(), total=len(df_fuel)):
        if i < (len(df_fuel)-n):
            x = df_fuel.rV.values[i:i+n]
            y = [i for i in range(i,i+n)]
            slope, intercept, _, _, _ = linregress(x, y)

        if slope < 0:
            ts_1 = row.ts
            ts_2 = None
            for j in range(i, len(df_fuel)):
                x_ = df_fuel.rV.values[j:j+n]
                y_ = [i for i in range(j,j+n)]
                slope_, intercept_, _, _, _ = linregress(x_, y_)
                if slope_ >= -0.1:
                    ts_2 = df_fuel.loc[j, 'ts']
                    break

        f_df = df_fuel[(df_fuel['ts']<=ts_2) & (df_fuel['ts']>=ts_1)]
        f_delta = f_df.rV.values[0] - f_df.rV.values[-1]

        e_flag = False
        for ig_s, ig_e in ignition_list:
            if (ig_s <= ts_2) and (ig_e >= ts_1):
                e_flag = True
                break 

        if f_delta>2 and not e_flag:
            date_dict['index'].append(i)
            date_dict['ts'].append(ts_1)

    date_df = pd.DataFrame(date_dict).set_index('index')

    df_list = [d for _, d in date_df.groupby(date_df.index - np.arange(len(date_df)))]
    for df in df_list:
        try:
            df = df.reset_index(drop=True)
        except Exception as e:
            pass
        
        if len(df) > 1:
            d_start = df.loc[0, 'ts']
            d_end = df.loc[len(df)-1, 'ts'] + de
            fuel_consumed_df = df_fuel[(df_fuel['ts']<=d_end) & (df_fuel['ts']>=d_start)]
            fuel_consumed = fuel_consumed_df.rV.values[0] - fuel_consumed_df.rV.values[-1]
            # print(d_end, d_start, fuel_consumed)

            output_df['start_ts'].append(d_start)
            output_df['end_ts'].append(d_end)
            output_df['duration_drained'].append(str(d_end-d_start))
            output_df['fuel_drained'].append(fuel_consumed)

    print("Saving output...")
    pd.DataFrame(output_df).to_csv(BASE_DIR+'draining_output.csv', index=False)
    print("Output saved.")

def detect_fuel_drain_ind(df_fuel, name='veh', delta_f=3, delta_t=timedelta(minutes=40)):
    output_df = {
        'veh': [],
        'termid': [],
        'start_ts': [],
        'end_ts': [],
        'duration_drained': [],
        'fuel_change': [],
        'CPH': []
    }

    # n = 2
    de = pd.Timedelta('2 minutes')
    win = delta_t
    date_dict = {
        'index': [],
        'ts': []
    }
    # sign_flags = []

    df_fuel.rV = df_fuel.rV.rolling(3, min_periods=1).mean()

    for i, row in tqdm(df_fuel.iterrows(), total=len(df_fuel)):
        if i < (len(df_fuel)):
            ts_i = row.ts
            ts_i_end = row.ts + win
            # x = df_fuel.rV.values[i:i+n]
            fuel_i = df_fuel[(df_fuel['ts'] >= ts_i) & (df_fuel['ts'] <= ts_i_end)]
            y = fuel_i.rV.values
            x = date2num(fuel_i.ts.values)
            slope, intercept, _, _, _ = linregress(x, y)

            ts_1 = None
            ts_2 = None

            if slope < -0.1:
                ts_1 = row.ts
                ts_2 = None
                for j in range(i, len(df_fuel)):
                    ts_j = df_fuel.loc[j, 'ts']
                    ts_j_end = ts_j + win
                    fuel_j = df_fuel[(df_fuel['ts'] >= ts_j) & (df_fuel['ts'] <= ts_j_end)]
                    y_ = fuel_j.rV.values
                    x_ = date2num(fuel_j.ts.values)
                    slope_, intercept_, _, _, _ = linregress(x_, y_)
                    if slope_ >= -0.1:
                        ts_2 = df_fuel.loc[j, 'ts']
                        break 
            elif slope > 0.1:
                ts_1 = row.ts
                ts_2 = None
                for j in range(i, len(df_fuel)):
                    ts_j = df_fuel.loc[j, 'ts']
                    ts_j_end = ts_j + win
                    fuel_j = df_fuel[(df_fuel['ts'] >= ts_j) & (df_fuel['ts'] <= ts_j_end)]

                    y_ = fuel_j.rV.values
                    x_ = date2num(fuel_j.ts.values)
                    slope_, intercept_, _, _, _ = linregress(x_, y_)
                    if slope_ <= 0.1:
                        ts_2 = df_fuel.loc[j, 'ts']
                        break
                    
            if ts_1!=None and ts_2!=None:
                f_df = df_fuel[(df_fuel['ts']<=ts_2) & (df_fuel['ts']>=ts_1)]

                f_delta = 0
                if len(f_df):
                    f_delta = f_df.rV.values[0] - f_df.rV.values[-1]

                e_flag = False
                # for ig_s, ig_e in ignition_list:
                #     if (ig_s <= ts_2) and (ig_e >= ts_1):
                #         e_flag = True
                #         break 

                if f_delta>5 and not e_flag:
                    date_dict['index'].append(i)
                    date_dict['ts'].append(ts_1)

    date_df = pd.DataFrame(date_dict).set_index('index')

    df_list = [d for _, d in date_df.groupby(date_df.index - np.arange(len(date_df)))]
    for df in df_list:
        try:
            df = df.reset_index(drop=True)
        except Exception as e:
            pass
        
        if len(df) > 1:
            d_start = df.loc[0, 'ts']
            d_end = df.loc[len(df)-1, 'ts'] + de
            fuel_consumed_df = df_fuel[(df_fuel['ts']<=d_end) & (df_fuel['ts']>=d_start)]
            fuel_consumed = fuel_consumed_df.rV.values[-1] - fuel_consumed_df.rV.values[0]
            # print(d_end, d_start, fuel_consumed)

            if abs(fuel_consumed) > delta_f:
                output_df['veh'].append(fuel_consumed_df.veh.values[0])
                output_df['termid'].append(fuel_consumed_df.termid.values[0])
                output_df['start_ts'].append(d_start)
                output_df['end_ts'].append(d_end)
                output_df['duration_drained'].append(str(d_end-d_start))
                output_df['fuel_change'].append(fuel_consumed)
                output_df['CPH'].append(fuel_consumed/((d_end-d_start).total_seconds()/3600))

    # print("Saving output...")
    df_out = pd.DataFrame(output_df)
    return df_out
    # print("Output saved.")
 