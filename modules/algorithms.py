from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import linregress

from matplotlib.dates import date2num

from .config import BASE_DIR, VEHICLE_DIR

# test comment by AA

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
            ign_df = df_elm[(df_elm['ts'] >= ts_1) & (df_elm['ts'] <= ts_2)]
            idl_df = ign_df[ign_df['VR'] < 1.0]

            list_of_df = [d for _, d in idl_df.groupby(
                idl_df.index - np.arange(len(idl_df)))]

            for idl in list_of_df:
                try:
                    idl = idl.reset_index(drop=True)
                except Exception as e:
                    pass

                i_start = idl.loc[0, 'ts']
                i_end = idl.loc[len(idl)-1, 'ts']
                if len(idl) > 1 and (i_end - i_start) > dl:
                    fuel_cdf = df_fuel[(df_fuel['ts'] >= i_start) & (
                        df_fuel['ts'] <= i_end)]
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
            y = [i for i in range(i, i+n)]
            slope, intercept, _, _, _ = linregress(x, y)

        if slope < 0:
            ts_1 = row.ts
            ts_2 = None
            for j in range(i, len(df_fuel)):
                x_ = df_fuel.rV.values[j:j+n]
                y_ = [i for i in range(j, j+n)]
                slope_, intercept_, _, _, _ = linregress(x_, y_)
                if slope_ >= -0.1:
                    ts_2 = df_fuel.loc[j, 'ts']
                    break

        f_df = df_fuel[(df_fuel['ts'] <= ts_2) & (df_fuel['ts'] >= ts_1)]
        f_delta = f_df.rV.values[0] - f_df.rV.values[-1]

        e_flag = False
        for ig_s, ig_e in ignition_list:
            if (ig_s <= ts_2) and (ig_e >= ts_1):
                e_flag = True
                break

        if f_delta > 2 and not e_flag:
            date_dict['index'].append(i)
            date_dict['ts'].append(ts_1)

    date_df = pd.DataFrame(date_dict).set_index('index')

    df_list = [d for _, d in date_df.groupby(
        date_df.index - np.arange(len(date_df)))]
    for df in df_list:
        try:
            df = df.reset_index(drop=True)
        except Exception as e:
            pass

        if len(df) > 1:
            d_start = df.loc[0, 'ts']
            d_end = df.loc[len(df)-1, 'ts'] + de
            fuel_consumed_df = df_fuel[(df_fuel['ts'] <= d_end) & (
                df_fuel['ts'] >= d_start)]
            fuel_consumed = fuel_consumed_df.rV.values[0] - \
                fuel_consumed_df.rV.values[-1]
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
            fuel_i = df_fuel[(df_fuel['ts'] >= ts_i) &
                             (df_fuel['ts'] <= ts_i_end)]
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
                    fuel_j = df_fuel[(df_fuel['ts'] >= ts_j)
                                     & (df_fuel['ts'] <= ts_j_end)]
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
                    fuel_j = df_fuel[(df_fuel['ts'] >= ts_j)
                                     & (df_fuel['ts'] <= ts_j_end)]

                    y_ = fuel_j.rV.values
                    x_ = date2num(fuel_j.ts.values)
                    slope_, intercept_, _, _, _ = linregress(x_, y_)
                    if slope_ <= 0.1:
                        ts_2 = df_fuel.loc[j, 'ts']
                        break

            if ts_1 != None and ts_2 != None:
                f_df = df_fuel[(df_fuel['ts'] <= ts_2) &
                               (df_fuel['ts'] >= ts_1)]

                f_delta = 0
                if len(f_df):
                    f_delta = f_df.rV.values[0] - f_df.rV.values[-1]

                e_flag = False
                # for ig_s, ig_e in ignition_list:
                #     if (ig_s <= ts_2) and (ig_e >= ts_1):
                #         e_flag = True
                #         break

                if f_delta > 5 and not e_flag:
                    date_dict['index'].append(i)
                    date_dict['ts'].append(ts_1)

    date_df = pd.DataFrame(date_dict).set_index('index')

    df_list = [d for _, d in date_df.groupby(
        date_df.index - np.arange(len(date_df)))]
    for df in df_list:
        try:
            df = df.reset_index(drop=True)
        except Exception as e:
            pass

        if len(df) > 1:
            d_start = df.loc[0, 'ts']
            d_end = df.loc[len(df)-1, 'ts'] + de
            fuel_consumed_df = df_fuel[(df_fuel['ts'] <= d_end) & (
                df_fuel['ts'] >= d_start)]
            fuel_consumed = fuel_consumed_df.rV.values[-1] - \
                fuel_consumed_df.rV.values[0]
            # print(d_end, d_start, fuel_consumed)

            if abs(fuel_consumed) > delta_f:
                output_df['veh'].append(fuel_consumed_df.veh.values[0])
                output_df['termid'].append(fuel_consumed_df.termid.values[0])
                output_df['start_ts'].append(d_start)
                output_df['end_ts'].append(d_end)
                output_df['duration_drained'].append(str(d_end-d_start))
                output_df['fuel_change'].append(fuel_consumed)
                output_df['CPH'].append(
                    fuel_consumed/((d_end-d_start).total_seconds()/3600))

    # print("Saving output...")
    df_out = pd.DataFrame(output_df)
    return df_out
    # print("Output saved.")


def generate_fuel_report(df_fuel, df_event):
    output_df = {
        "date": [],
        "opertion time": [],
        "operation time %": [],
        "Initial Vol.": [],
        "Final Vol.": [],
        "Net Fuel Consumption": [],
        "Refuelling Volume": [],
        "Raw Fuel Consumption": [],
        "Flag": []
    }
    df_fuel = df_fuel.sort_values(by='ts')
    # df_fuel.rV = df_fuel.rV.rolling(2, min_periods=1).mean()
    df_event = df_event.sort_values(by='ts')

    df_fuel['date'] = df_fuel.ts.dt.date.values
    df_event['date'] = df_event.ts.dt.date.values

    date_vals = np.unique(df_fuel.ts.dt.date.values)

    for o, d in enumerate(date_vals):
        temp_fuel = df_fuel[df_fuel['date'] == d].reset_index(drop=True)
        temp_event = df_event[df_event['date'] == d].reset_index(drop=True)

        output_df["date"].append(d)
        output_df["Initial Vol."].append(temp_fuel.loc[0, "rV"])
        output_df["Final Vol."].append(temp_fuel.loc[(len(temp_fuel)-1), "rV"])

        fuel_consumption = temp_fuel.loc[0, "rV"]
        refuel_vol = 0
        run_hours = pd.Timedelta("0")

        for i, row in temp_event.iterrows():
            if row.type == 5:
                # print("Row 5")
                ts_1 = row.ts
                ts_2 = None
                for j in range(i, len(temp_event)):
                    if temp_event.loc[j, 'type'] == 6:
                        # print("Row 6")
                        ts_2 = temp_event.loc[j, 'ts']
                        break

                if ts_2 != None:
                    temp_cons = temp_fuel[(temp_fuel['ts'] >= ts_1) & (temp_fuel['ts'] <= ts_2)].reset_index()
                else:
                    temp_cons = temp_fuel[(temp_fuel['ts'] >= ts_1)].reset_index()

                # print(f"\ni: {ts_1}, j: {ts_2}")
                if len(temp_cons) > 0:
                    # print(temp_cons)
                    fuel_consumption -= temp_cons.loc[0, 'rV'] - temp_cons.loc[len(temp_cons)-1, 'rV']
                    run_hours += temp_cons.loc[(len(temp_cons)-1), 'ts'] - temp_cons.loc[0, 'ts']

            elif row.type == 1:
                # print("Row 1")
                ts_1 = row.ts
                ts_2 = None
                for j in range(i, len(temp_event)):
                    if temp_event.loc[j, 'type'] == 2:
                        # print("Row 2")
                        ts_2 = temp_event.loc[j, 'ts']
                        break

                if ts_2 != None:
                    temp_cons = temp_fuel[(temp_fuel['ts'] >= ts_1) & (temp_fuel['ts'] <= ts_2)].reset_index()
                else:
                    temp_cons = temp_fuel[temp_fuel['ts'] >= ts_1].reset_index()
                
                # print(f"\ni: {ts_1}, j: {ts_2}")
                if len(temp_cons) > 0:
                    # print(temp_cons)
                    refuel_vol += temp_cons.loc[len(temp_cons)-1, 'rV'] - temp_cons.loc[0, 'rV'] 
                    # fuel_consumption += temp_cons.loc[len(temp_cons)-1, 'rV'] - temp_cons.loc[0, 'rV']

        if refuel_vol < 1:
            raw_fuel = abs(temp_fuel.loc[0, "rV"] - fuel_consumption)
            # raw_fuel = fuel_consumption
        else:
            raw_fuel = abs(abs(temp_fuel.loc[0, "rV"] + refuel_vol - temp_fuel.loc[len(temp_fuel)-1, "rV"]))
            # raw_fuel = fuel_consumption

        output_df['Raw Fuel Consumption'].append(raw_fuel)
        output_df['Refuelling Volume'].append(refuel_vol)

        s = run_hours.total_seconds()
        hours = s//3600
        s = s - (hours*3600)
        minutes = s//60
        seconds = s - (minutes*60)
        op_hours = '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))

        output_df['opertion time'].append(op_hours)
        output_df['operation time %'].append(((run_hours.total_seconds()/3600)/24)*100)
        net_fuel = temp_fuel.loc[0, "rV"] + refuel_vol - temp_fuel.loc[(len(temp_fuel)-1), "rV"]
        output_df['Net Fuel Consumption'].append(net_fuel)
        if abs(net_fuel - output_df['Raw Fuel Consumption'][o]) > 3:
            output_df['Flag'].append('High')
        else:
            output_df['Flag'].append(' ')

    return pd.DataFrame(output_df)

def check_time_diff(event_df, cph_df):
    ignition_list = []
    for i, row in event_df.iterrows():
        if row.type == 5:
            ts_1 = row.ts
            ts_2 = None
            for j in range(i, len(event_df)):
                if event_df.loc[j, 'type'] == 6:
                    ts_2 = event_df.loc[j, 'ts']
                    break
            ignition_list.append((ts_1, ts_2))

    ignition_list_cph = []
    for i, row in cph_df.iterrows():
        ignition_list_cph.append((row.Ign_START_IST, row.Ign_STOP_IST))

    ign_faulty_pairs = []
    for ign_pair_c  in ignition_list_cph:
        flg = True
        for ign_pair in ignition_list:
            if (ign_pair[0]<=ign_pair_c[1]) and (ign_pair[1]>=ign_pair_c[0]):
                flg = False
                break
        if flg:
            ign_faulty_pairs.append(ign_pair_c)

    data = {
        'Ign_START': [a for a,b in ign_faulty_pairs],
        'Ign_STOP': [b for a,b in ign_faulty_pairs]
    }

    return pd.DataFrame(data)

