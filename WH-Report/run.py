import numpy as np
import pandas as pd
from argparse import ArgumentParser
from datetime import datetime, date, timedelta
from Modules.algorithms import check_midnight_bug, generate_wh_report
from Modules.config import IST, SITE_CODES, SITE_REGISTER, UTC
import time
import pytz
from tqdm import tqdm
from pyreadr import read_r
import os

import warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser()

parser.add_argument("-s", "--site", nargs='+', help="The name of site: {bagru1, bagru2, dand, jobner, sawarda}", default=['bagru1'])
parser.add_argument("--start", type=str, help="The starting date for data")
parser.add_argument("--end", type=str, help="The ending date for data")
parser.add_argument("--elm_file", type=str, help="The file path for raw ELM data")
parser.add_argument("--fuel_file", type=str, help="The file path for raw fuel data")
parser.add_argument("--event_file", type=str, help="The file path for raw event data")
parser.add_argument("--output_file", type=str, help="The file path for report output")

parser.add_argument("--delta_t", type=int, help="Threshold in minutes for detecting missing hrs", default=30)
parser.add_argument("--agg_t", help="The aggregration frequency for data preparation", default=1)

args = parser.parse_args()

if os.path.exists(args.fuel_file):
    pass
else:
    print(f"Fuel file {args.fuel_file} does not exist.")
    raise FileNotFoundError

if os.path.exists(args.event_file):
    pass
else:
    print(f"Event file {args.event_file} does not exist.")
    raise FileNotFoundError

d1 = [int(i) for i in args.start.split('-')]
d2 = [int(i) for i in args.end.split('-')]

start_datetime = datetime(d1[0], d1[1], d1[2], 0, 0, 0, 0, IST)
end_datetime = datetime(d2[0], d2[1], d2[2], 23, 59, 59, 0, IST)

# Preparing unique datelist
date_list = []
d = start_datetime.date()
while(d<=end_datetime.date()):
    date_list.append(d)
    d += timedelta(days=1)

print(f"Reading raw ELM data from {args.elm_file} ...")
st  = time.time()
elm_file_base = args.elm_file.split('/')[:-1]
elm_file_reg = args.elm_file.split('/')[-1]

df_main_list = []
for dt in date_list:
    try:
        if '.RDS' in args.elm_file:
            rds_main = read_r("/".join(elm_file_base) + '/' + elm_file_reg.replace("*", str(dt)))
            df_main = rds_main[None]
        elif '.csv' in args.elm_file:
            df_main = pd.read_csv("/".join(elm_file_base) + '/' + elm_file_reg.replace("*", str(dt)))
        df_main_list.append(df_main)
    except Exception as e:
        print(e)

df_main = pd.concat(df_main_list).reset_index(drop=True)


# if '.RDS' in args.elm_file:
#     rds_main = read_r(args.elm_file)
#     df_main = rds_main[None]
# elif '.csv' in args.elm_file:
#     df_main = pd.read_csv(args.elm_file)


print(f"File {args.elm_file} read (took {time.time()-st:.2f}s).")

de = pd.Timedelta(f"{args.delta_t} minutes")
site_code = SITE_CODES[args.site[0]]

print("Converting and localizing datetime...")
st  = time.time()
# Converting  to datetime
df_main[f'ts'] = pd.to_datetime(df_main[f'ts'], utc=True)
df_main[f'ts'] = df_main[f'ts'].apply(lambda x: x.astimezone(IST))


print(f"Datetime processed (took {time.time()-st:.2f}s).")
print(df_main)


# Removing preliminary missing values
df_main = df_main.dropna(subset=[f'VR'])
df_main = df_main.reset_index(drop=True)

print(f"Aggregating data to {args.agg_t} minutes frequency...")
st  = time.time()
# Aggregating Parameters
params_list = []
df_main.set_index('ts', inplace=True)

param_info = SITE_REGISTER[SITE_REGISTER['site']==args.site[0]]

groups = [x for _, x in param_info.groupby('params')]

df_dict = {
    'ts': [],
    'acwatts': [],
    'ebwatts': [],
    'dgwatts': [],
    'ibattTOT': [],
    'dcv': []
}
df_dict_trans = {
    'acwatts': [],
    'ebwatts': [],
    'dgwatts': [],
    'ibattTOT': [],
    'dcv': []
}

p = []
for g in groups:
    for i, row in g.iterrows():
        r = row.registers.split("|")
        for reg in r:
            val_df =df_main[(df_main['topic']==f'{site_code}/{row.meter}') & (df_main['ST']==int(reg))].resample(f"{args.agg_t}T").mean()
            p.append((g['params'].values[0], val_df))
            df_dict_trans[g['params'].values[0]].append(val_df)

base_date = pd.Timestamp("2100-01-01")

shortest =  sorted(p, key=lambda x: (x[1].index.values[-1], base_date-x[1].index.values[0]))[0][1]
shortest = shortest.reset_index()

for key, values in df_dict_trans.items():
    tot = []
    for value in values:
        v = value.loc[shortest.iloc[0].ts: shortest.iloc[-1].ts].VR.values
        tot.append(v)
    tot = np.array(tot)

    df_dict[key] = np.sum(tot, axis=0)

df_dict['ts'] = shortest['ts'].values

df = pd.DataFrame(df_dict)

print(f"Dataframe aggregated from raw ELM data (took {time.time()-st:.2f}s).")

# Localizing timestamps
print("Localizing timestamps...")
st = time.time()
df.ts = df.ts.apply(lambda x: x.tz_localize(UTC))
df.ts = df.ts.apply(lambda x: x.astimezone(IST))
print(f"Done localizing timestamps (took {time.time()-st:.2f}s).")
print(df)

# Getting unix timestamps and differences
print("Getting unix stamp difference for integration...")
st = time.time()
start_time = datetime(2021, 9, 30, 18, 30, 0, 0, UTC)
start_unix = time.mktime(start_time.timetuple())

unixStamps = []
for time_val in tqdm(df[f'ts'].values):
    unixStamps.append((time_val - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))

df['unixStamps'] = unixStamps
df['unixStampsDiff'] = (df.unixStamps.values - start_unix)/3600
print(f"Done adding unix stamp difference (took {time.time()-st:.2f}s).")

f_name = args.event_file
print(f"Fetching event data from {f_name} ...")
st = time.time()
if '.RDS' in args.fuel_file:
    rds_event = read_r(f_name)
    df_event = rds_event[None]
elif '.csv' in args.fuel_file:
    df_event = pd.read_csv(args.event_file)

df_event.ts = pd.to_datetime(df_event.ts, utc=True)

df_event[f'ts'] = df_event[f'ts'].apply(lambda x: x.astimezone(IST)) 

df_event = df_event[(df_event['ts']>=start_datetime) & (df_event['ts']<=end_datetime)].reset_index(drop=True)

print(f"Fetched event data (took {time.time()-st:.2f}s).")

st = time.time()
print(f"Processing event data ...")
unixStamps = []
for time_val in df_event[f'ts'].values:
    unixStamps.append((time_val - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))

df_event['unixStamps'] = unixStamps

print(df_event)

ignition_list = []
for i, row in tqdm(df_event.iterrows()):
    if row['type'] == 5:
        ts_1 = row.unixStamps
        ts_2 = None
        for j in range(i, len(df_event)):
            if df_event.loc[j, 'type'] == 6:
                ts_2 = df_event.loc[j, 'unixStamps']
                break
        ignition_list.append((ts_1, ts_2))

    else:
      ignition_list.append(None)

df_event['ignition_list'] = ignition_list

refuel_list = []
for i, row in tqdm(df_event.iterrows()):
    if row['type'] == 1:
        ts_1 = row.unixStamps
        ts_2 = None
        for j in range(i, len(df_event)):
            if df_event.loc[j, 'type'] == 2:
                ts_2 = df_event.loc[j, 'unixStamps']
                break
        refuel_list.append((ts_1, ts_2))

    else:
      refuel_list.append(None)

df_event['refuel_list'] = refuel_list
df_event['ignition_list'] = ignition_list
df_event['date'] = df_event.ts.dt.date.values

print(f"Processed event data (took {time.time()-st:.2f}s).")
print(df_event[df_event['type']==5])
print(df_event[df_event['type']==1])

df = df.dropna(subset=[f'acwatts'])
df = df.reset_index(drop=True)
df = df.fillna(0)

print(f"Reading fuel data from {args.fuel_file}...")
st = time.time()

if '.RDS' in args.fuel_file:
    df_fuel = read_r(args.fuel_file)[None]
elif '.csv' in args.fuel_file:
    df_fuel = pd.read_csv(args.fuel_file)
print(f"Read fuel data (took {time.time()-st:.2f}s).")
print(df_fuel)

print(f"Processing fuel data...")
st = time.time()

df_fuel.ts = pd.to_datetime(df_fuel.ts, utc=True)
df_fuel.ts = df_fuel.ts.apply(lambda x: x.astimezone(IST))
df_fuel["date"] = df_fuel.ts.dt.date.values

unixStamps = []
for time_val in df_fuel[f'ts'].values:
    unixStamps.append((time_val - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))

df_fuel['unixStamps'] = unixStamps
print(f"Processed fuel data (took {time.time()-st:.2f}s).")
print(df_fuel)

m = start_datetime.strftime("%b")
# s_name = f"{SITE_CODES[args.site[0]]}_{m}_report_sup.csv"
s_name = args.output_file
print(f"Generating and saving report to {s_name} ...")
output_df = generate_wh_report(df, df_event, df_fuel, date_list, SITE_CODES[args.site[0]], de)
output_df.to_csv(s_name)
print(f"DONE!")