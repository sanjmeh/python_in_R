import numpy as np
from datetime import datetime, time, date
import pandas as pd
from .config import IST, STRUCT, WEEK_DAYS


def get_missing_stamps(df, d, de=pd.Timedelta("30 minutes")):
  '''
  To return missing timestamps for a day

  Parameters:
  df (DataFrame): Aggregated dataset for the day
  d (Date): date
  de (Timedelta): Threshold time difference for missing data

  Returns:
  time_stamps (List): A list of tuples containg all ignition start and stop times for day d 
  '''

  # initialising start and end times to check for boundary misses
  start_time = IST.localize(datetime.combine(d, time(0, 0, 0)))
  end_time = IST.localize(datetime.combine(d, time(23, 59, 59)))

  # Initialising empty list
  time_stamps = []

  # Iterating through the DataFrame df
  for i, row in df.iterrows():
    # checking for initial boundary data miss
    if i == 0:
      cons_diff = abs(df.loc[i, f'ts'] - start_time)
      if cons_diff > de:
        time_stamps.append((start_time, row[f'ts']))

    # checking for intermediate data miss
    elif i < (len(df)-1):
      cons_diff = abs(df.loc[i+1, f'ts'] - df.loc[i, f'ts'])
      if cons_diff > de:
        time_stamps.append((row[f'ts'], df.loc[i+1, f'ts']))

    # checking for final boundary data miss
    else:
      cons_diff = abs(end_time - df.loc[i, f'ts'])
      if cons_diff > de:
        time_stamps.append((row[f'ts'], end_time))

  return time_stamps


def get_missing_hrs(df, d, de=pd.Timedelta("30 minutes")):
  '''
  To return missing hours and timestamps for a day

  Parameters:
  df (DataFrame): Aggregated dataset for the day
  d (Date): date
  de (Timedelta): Threshold time difference for missing data

  Returns:
  missing_hrs (Float): Missing hours in a day
  time_stamps (List): A list of tuples containg all ignition start and stop times for day d 
  '''
  # Getting missing time stamps for the day
  time_stamps = get_missing_stamps(df, d, de)
  secs = 0

  # Summing up the timestamp differences
  for s, e in time_stamps:
    secs += (e-s).total_seconds()

  return secs/3600, time_stamps


def get_median_wh(df, date_list, baseline, topline):
  '''
  To return median Wh and stddev for a given duration during the day

  Parameters:
  df (DataFrame): Aggregated dataset for the day
  date_list (List): List of all dates in consideration
  baseline (Time): Starting time for duration
  topline (Time): Ending time for duration

  Returns:
  median_wh (Float): Median Wh for the duration on rest of the days
  wh_stddev (Float): Standar deviation of Wh for the duration on rest of the day 
  '''

  df['time'] = df[f'ts'].dt.time.values
  df['date'] = df[f'ts'].dt.date.values

  df_temp = df[(df['time'] > baseline) & (df['time'] < topline)].reset_index(drop=True)
  output_df = {
    'date': [],
    'wh': [],
    'watts': []
  }

  for d in date_list:
    dft = df_temp[df_temp['date'] == d]
    _time = 0

    wt_list = []

    try:
      ac_df = dft[dft[f'acwatts'] > 100]
      ac_df_list = [d for _, d in ac_df.groupby(
        ac_df.index - np.arange(len(ac_df)))]
    except:
      print(f"AC issue: {d}")
      ac_df_list = []

    for ac_df in ac_df_list:
      x = ac_df['unixStampsDiff'].values
      y = ac_df[f'acwatts'].values

      wh = np.trapz(y, x)

      _time += wh
      wt_list = np.mean(y)

    output_df['date'].append(d)
    output_df['wh'].append(_time)
    output_df['watts'].append(np.mean(np.array(wt_list)))

  output_df = pd.DataFrame(output_df)

  return output_df['watts'].median(), output_df['watts'].std()


def hrs_in_boundary(st, ed):
  '''
  To check if missing data is at the start/end of the day

  Parameters:
  st: starting timestamp of missing data
  ed: ending timestamp of missing data

  Returns:
  boundary_flag (Boolean): True if missing data is at the start/end of the day else False
  '''
  b1 = time(0, 10, 0)
  b2 = time(23, 50, 0)

  if b1 <= ed.time() and b1 >= st.time():
    return True
  if b2 <= ed.time() and b2 >= st.time():
    return True

  return False

def check_midnight_bug(df):
  for i in range(1, 31):
    f = df[((df['type']==5)|(df['type']==6))&(df['date']==date(2021,10,i))]
    try:
      if f['type'].iloc[0] != 5 or f['type'].iloc[-1] != 6:
        print(date(2021,10,i))
    except Exception as e:
      # print(f"error {e}")
      pass

def generate_wh_report(df, df_event, df_fuel, date_vals, site_code, de=pd.Timedelta("30 minutes")):
  '''
  Function to generate a report having structure defined in struct.txt

  Parameters:
  df (DataFrame): Raw ELM data for the site
  df_event (DataFrame): Event data for the site
  date_vals (List): List of dates for the data in question
  site_code (Str): One of {bagru1, bagru2, dand, jobner001, saw001} used for querying data
  de (Timedelta): Threshold time difference for missing data

  Returns:
  output_df (DataFrame): A DataFrame structured as struct.txt containing the final report
  '''

  # Defining an empty dictionary and initialising all key (columns) to empty list
  output_df = {}
  for col in STRUCT:
    output_df[col] = []

  # Create a date column to filter data easily
  df[STRUCT[0]] = df[f'ts'].dt.date.values

  # Iterating through all dates to get data for individual days
  for o, d in enumerate(date_vals):

    # Filtering data for a day and resetting index to avoid errors while querying
    temp_df = df[df[STRUCT[0]] == d].reset_index(drop=True)
    temp_df_event = df_event[df_event[STRUCT[0]] == d].reset_index(drop=True)
    temp_df_fuel = df_fuel[df_fuel[STRUCT[0]] == d].reset_index(drop=True)

    # Getting missing hours and missing timestamps for the day
    missing_hrs, missing_stamps = get_missing_hrs(temp_df, d, de)

    # Initialising boundary_flag as False and checking by iterating over missing timestamps
    boundary_flag = False

    # Getting the median wh in missing timestamps for other days
    median_wh = 0
    for stamp in missing_stamps:
      med, stddev = get_median_wh(df, date_vals, stamp[0].time(), stamp[1].time())
      median_wh += med
      boundary_flag = hrs_in_boundary(stamp[0], stamp[1])

    # Averaging the median run hours for all missing timestamps to get median_wh
    try:
      median_wh = median_wh/len(missing_stamps)
    except:
      pass
    
    # Adding the date and day of the week to the dataframe
    output_df[STRUCT[0]].append(d)
    output_df[STRUCT[32]].append(site_code)
    output_df[STRUCT[33]].append(d.strftime("%B"))
    output_df[STRUCT[1]].append(WEEK_DAYS[d.weekday()])

    # Setting x and y values for calculating auc for 101, 103+105 and 107
    x = temp_df['unixStampsDiff'].values
    y = temp_df[f'acwatts'].values
    y_eb = temp_df[f'ebwatts'].values
    y_dg = temp_df[f'dgwatts'].values

    ebwh = np.trapz(y_eb, x)
    dgwh = np.trapz(y_dg, x)
    twh = np.trapz(y, x)

    # Adding auc for 101, 103+105 & 107, and missing hours to the dataframe
    output_df[STRUCT[4]].append(ebwh)
    output_df[STRUCT[12]].append(dgwh)
    output_df[STRUCT[23]].append(twh)
    output_df[STRUCT[3]].append(missing_hrs)

    # Initialising variables to 0 for calculating various params as described below
    eb_time = 0
    dg_time = 0
    dg_wh = 0
    eb_wh = 0
    bat_char_time = 0
    bat_char_time2 = 0
    bat_dischar_time = 0
    bat_dischar_time2 = 0
    ac_zero_time = 0
    ac_100_time = 0
    dg_ign_time = 0
    refuel = 0
    cons_ign = 0

    # filtering all start events for the day
    event_5 = temp_df_event[(temp_df_event['type'] == 5)]
    event_1 = temp_df_event[(temp_df_event['type'] == 1)]
    
    try:
      initial_vol = temp_df_fuel.loc[0, 'rV']
      final_vol = temp_df_fuel.loc[len(temp_df_fuel)-1, 'rV']
    except:
      initial_vol =  -1
      final_vol = -1

    # Iterating over filtered events to calculate dg ignition time and ignition based dg Wh
    for evi, evrow in event_5.iterrows():
      try:
        dq = datetime.fromtimestamp(evrow.ignition_list[1])
        dq_ist = dq.astimezone(IST)
        if dq_ist.date() == d:
          dg_ign_time += evrow.ignition_list[1] - evrow.ignition_list[0]
        else:
          q = datetime.combine(d, time(18, 30, 0))
          q = (q - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
          dg_ign_time += q - evrow.ignition_list[0]

        dg_wh_df = temp_df[(temp_df['unixStamps'] <= evrow.ignition_list[1]) & (temp_df['unixStamps'] >= evrow.ignition_list[0])]


        x_dg = dg_wh_df['unixStampsDiff'].values
        y_dg = dg_wh_df[f'acwatts'].values

        dgwh = np.trapz(y_dg, x_dg)

        dg_wh += dgwh

        try:
          cons_df = temp_df_fuel[(temp_df_fuel['unixStamps'] <= (evrow.ignition_list[1]+60)) & (temp_df_fuel['unixStamps'] >= (evrow.ignition_list[0]-60))].reset_index(drop=True)
        except Exception as e:
          print(f"{e} {evrow.ignition_list}")
          cons_df = temp_df_fuel[temp_df_fuel['unixStamps'] >= (evrow.ignition_list[0]-60)].reset_index(drop=True)

        if len(cons_df) > 0:
          cons_ign += cons_df.loc[0, 'rV'] - cons_df.loc[len(cons_df)-1, 'rV']

      except Exception as e:
        print(f"event 5: {e} {evrow.ignition_list}")
    
    # Iterating over filtered events to calculate refuelling
    for evi, evrow in event_1.iterrows():
      try:
        try:
          refuel_df = temp_df_fuel[(temp_df_fuel['unixStamps'] <= evrow.refuel_list[1]) & (temp_df_fuel['unixStamps'] >= evrow.refuel_list[0])].reset_index(drop=True)
        except Exception as e:
          print(f"{e} {evrow.refuel_list}")
          refuel_df = temp_df_fuel[temp_df_fuel['unixStamps'] >= evrow.refuel_list[0]].reset_index(drop=True)

        if len(refuel_df) > 0:
          refuel += refuel_df.loc[len(refuel_df)-1, 'rV'] - refuel_df.loc[0, 'rV']


      except Exception as e:
        print(f"event 1: {e}")

    try:
      _, e_df2 = [x for _, x in temp_df.groupby(temp_df[f'ebwatts'] > 100)]
      e_df2_list = [d for _, d in e_df2.groupby(e_df2.index - np.arange(len(e_df2)))]
    except:
      print(f"EB issue: {d}")
      e_df2_list = []

    try:
      _, a_df2 = [x for _, x in temp_df.groupby(temp_df[f'acwatts'] < 2)]
      a_df2_list = [d for _, d in a_df2.groupby(a_df2.index - np.arange(len(a_df2)))]
    except:
      print(f"AC issue: {d}")
      a_df2_list = []

    try:
      _, ac_df2 = [x for _, x in temp_df.groupby(temp_df[f'acwatts'] < 100)]
      ac_df2_list = [d for _, d in ac_df2.groupby(ac_df2.index - np.arange(len(ac_df2)))]
    except:
      print(f"AC issue: {d}")
      ac_df2_list = []

    try:
      _, d_df2 = [x for _, x in temp_df.groupby(temp_df[f'dgwatts'] > 100)]
      d_df2_list = [d for _, d in d_df2.groupby(d_df2.index - np.arange(len(d_df2)))]
    except:
      print(f"DG issue: {d}")
      d_df2_list = []

    try:
      _, c_df2 = [x for _, x in temp_df.groupby(temp_df[f'ibattTOT'] > 20)]
      c_df2_list = [d for _, d in c_df2.groupby(c_df2.index - np.arange(len(c_df2)))]
    except:
      print(f"Charge issue: {d}")
      c_df2_list = []

    try:
      _, di_df2 = [x for _, x in temp_df.groupby(temp_df[f'ibattTOT'] < -20)]
      di_df2_list = [d for _, d in di_df2.groupby(di_df2.index - np.arange(len(di_df2)))]
    except:
      print(f"Discharge issue: {d}")
      di_df2_list = []

# ################################################
    try:
      _, ch_df2 = [x for _, x in temp_df.groupby((temp_df[f'dcv'] < 51) & (temp_df[f'acwatts'] > 3000))]
      ch_df2_list = [d for _, d in ch_df2.groupby(ch_df2.index - np.arange(len(ch_df2)))]
    except:
      print(f"Charge issue: {d}")
      ch_df2_list = []

    try:
      _, dis_df2 = [x for _, x in temp_df.groupby((temp_df[f'dcv'] < 51) & (temp_df[f'acwatts'] < 4000))]
      dis_df2_list = [d for _, d in dis_df2.groupby(dis_df2.index - np.arange(len(dis_df2)))]
    except:
      print(f"Discharge issue: {d}")
      dis_df2_list = []

##################################################

    for e_df in e_df2_list:
      eb_time += e_df['unixStamps'].iloc[-1] - e_df['unixStamps'].iloc[0]

    for a_df in a_df2_list:
      ac_zero_time += a_df['unixStamps'].iloc[-1] - a_df['unixStamps'].iloc[0]

    for d_df in d_df2_list:
      dg_time += d_df['unixStamps'].iloc[-1] - d_df['unixStamps'].iloc[0]

    for c_df in c_df2_list:
      bat_char_time += c_df['unixStamps'].iloc[-1] - c_df['unixStamps'].iloc[0]

    for di_df in di_df2_list:
      bat_dischar_time += di_df['unixStamps'].iloc[-1] - di_df['unixStamps'].iloc[0]

    ########################################################

    for ch_df in ch_df2_list:
      bat_char_time2 += ch_df['unixStamps'].iloc[-1] - ch_df['unixStamps'].iloc[0]

    for dis_df in dis_df2_list:
      bat_dischar_time2 += dis_df['unixStamps'].iloc[-1] - dis_df['unixStamps'].iloc[0]

    #########################################################

    for ac_df in ac_df2_list:
      ac_100_time += ac_df['unixStamps'].iloc[-1] - ac_df['unixStamps'].iloc[0]

    projected_dg_hrs = (dg_ign_time - dg_time)/3600 if (((dg_ign_time - dg_time)/3600 > 0.2) and (missing_hrs > 0)) else 0
    projected_eb_hrs = (missing_hrs - projected_dg_hrs) if missing_hrs > 0 else 0
    projected_dg_wh = projected_dg_hrs*median_wh
    projected_eb_wh = projected_eb_hrs*median_wh

    output_df[STRUCT[6]].append(projected_eb_hrs)
    output_df[STRUCT[7]].append(projected_eb_wh)
    output_df[STRUCT[14]].append(projected_dg_hrs)
    output_df[STRUCT[15]].append(projected_dg_wh)

    if missing_hrs == 0:
      output_df[STRUCT[2]].append(eb_time/3600)
    else:
      output_df[STRUCT[2]].append(23.99-missing_hrs-(dg_time/3600)-(ac_100_time/3600))
    output_df[STRUCT[9]].append(dg_time/3600)
    output_df[STRUCT[10]].append(dg_ign_time/3600)
    output_df[STRUCT[17]].append(bat_char_time/3600)
    output_df[STRUCT[19]].append(bat_dischar_time/3600)
    output_df[STRUCT[21]].append(ac_zero_time/3600)

    output_df[STRUCT[18]].append(bat_char_time2/3600)
    output_df[STRUCT[20]].append(bat_dischar_time2/3600)
    output_df[STRUCT[22]].append(ac_100_time/3600)

    output_df[STRUCT[11]].append((dg_time-dg_ign_time)/3600)

    output_df[STRUCT[13]].append(dg_wh)

    if site_code == "dand":
      if boundary_flag:
        print(f"Boundary: {d}")
        eb_wh = twh - dg_wh
      else:
        eb_wh = twh - dg_wh - projected_dg_wh - projected_eb_wh
    else:
      if boundary_flag:
        print(f"Boundary: {d}")
        eb_wh = ebwh
      else:
        eb_wh = twh - dg_wh - projected_dg_wh - projected_eb_wh

    output_df[STRUCT[5]].append(eb_wh)

    if dg_ign_time < dg_time:
      output_df[STRUCT[8]].append(ebwh + projected_eb_wh)
      output_df[STRUCT[16]].append(dgwh + projected_dg_wh)
    else:
      output_df[STRUCT[8]].append(eb_wh + projected_eb_wh)
      output_df[STRUCT[16]].append(dg_wh + projected_dg_wh)

    if missing_hrs == 0:
      if site_code == 'dand':
        output_df[STRUCT[24]].append(dg_wh + projected_dg_wh + projected_eb_wh + eb_wh)
      else:
        output_df[STRUCT[24]].append(twh)
    else:
      if dg_ign_time >= dg_time:
        output_df[STRUCT[24]].append(dg_wh + projected_dg_wh + projected_eb_wh + eb_wh)
      else:
        output_df[STRUCT[24]].append(dgwh + projected_dg_wh + projected_eb_wh + ebwh)


    consumption = initial_vol - final_vol + refuel

    if refuel > 1:
      cons_final = cons_ign
      output_df[STRUCT[27]].append(cons_ign)
      output_df[STRUCT[28]].append(consumption-cons_ign)
    else:
      cons_final = consumption
      output_df[STRUCT[27]].append(consumption)
      output_df[STRUCT[28]].append(0)

    

    if site_code == 'dand':
      try:
        cph= cons_final/(dg_ign_time/3600)
      except:
        cph = 0
      idling_time = 0
    else:
      if (dg_time-dg_ign_time)>0:
        idling_time = 0
        cph= cons_final/((dg_time-idling_time)/3600)
      else:
        idling_time = (dg_ign_time - dg_time - projected_dg_hrs*3600)
        cph= cons_final/((dg_ign_time-idling_time)/3600)

    output_df[STRUCT[25]].append(initial_vol)
    output_df[STRUCT[26]].append(final_vol)
    
    output_df[STRUCT[29]].append(refuel)
    output_df[STRUCT[30]].append(idling_time/3600)
    output_df[STRUCT[31]].append(cph)

  return pd.DataFrame(output_df)

# def generate_wh_report_dand(df, df_event, df_fuel, date_vals, site_code, de=pd.Timedelta("30 minutes")):
#   '''
#   Function to generate a report for Dand having structure defined in struct.txt

#   Parameters:
#   df (DataFrame): Raw ELM data for the site
#   df_event (DataFrame): Event data for the site
#   date_vals (List): List of dates for the data in question
#   site_code (Str): One of {bagru1, bagru2, dand, jobner001, saw001} used for querying data
#   de (Timedelta): Threshold time difference for missing data

#   Returns:
#   output_df (DataFrame): A DataFrame structured as struct.txt containing the final report
#   '''

#   # Defining an empty dictionary and initialising all key (columns) to empty list
#   output_df = {}
#   for col in STRUCT:
#     output_df[col] = []

#   # Create a date column to filter data easily
#   df[STRUCT[0]] = df[f'ts'].dt.date.values

#   # Iterating through all dates to get data for individual days
#   for o, d in enumerate(date_vals):

#     # Filtering data for a day and resetting index to avoid errors while querying
#     temp_df = df[df[STRUCT[0]] == d].reset_index(drop=True)
#     temp_df_event = df_event[df_event[STRUCT[0]] == d].reset_index(drop=True)
#     temp_df_fuel = df_fuel[df_fuel[STRUCT[0]] == d].reset_index(drop=True)

#     # Getting missing hours and missing timestamps for the day
#     missing_hrs, missing_stamps = get_missing_hrs(temp_df, d, de)

#     # Initialising boundary_flag as False and checking by iterating over missing timestamps
#     boundary_flag = False

#     # Getting the median wh in missing timestamps for other days
#     median_wh = 0
#     for stamp in missing_stamps:
#       med, stddev = get_median_wh(df, date_vals, stamp[0].time(), stamp[1].time())
#       median_wh += med
#       boundary_flag = hrs_in_boundary(stamp[0], stamp[1])

#     # Averaging the median run hours for all missing timestamps to get median_wh
#     try:
#       median_wh = median_wh/len(missing_stamps)
#     except:
#       pass
    
#     # Adding the date and day of the week to the dataframe
#     output_df[STRUCT[0]].append(d)
#     output_df[STRUCT[32]].append(site_code)
#     output_df[STRUCT[33]].append(d.strftime("%B"))
#     output_df[STRUCT[1]].append(WEEK_DAYS[d.weekday()])

#     # Setting x and y values for calculating auc for 101, 103+105 and 107
#     x = temp_df['unixStampsDiff'].values
#     y = temp_df[f'acwatts'].values
#     y_eb = temp_df[f'ebwatts'].values
#     y_dg = temp_df[f'dgwatts'].values

#     ebwh = np.trapz(y_eb, x)
#     dgwh = np.trapz(y_dg, x)
#     twh = np.trapz(y, x)

#     # Adding auc for 101, 103+105 & 107, and missing hours to the dataframe
#     output_df[STRUCT[4]].append(ebwh)
#     output_df[STRUCT[12]].append(dgwh)
#     output_df[STRUCT[23]].append(twh)
#     output_df[STRUCT[3]].append(missing_hrs)

#     # Initialising variables to 0 for calculating various params as described below
#     eb_time = 0
#     dg_time = 0
#     dg_wh = 0
#     eb_wh = 0
#     bat_char_time = 0
#     bat_char_time2 = 0
#     bat_dischar_time = 0
#     bat_dischar_time2 = 0
#     ac_zero_time = 0
#     ac_100_time = 0
#     dg_ign_time = 0
#     refuel = 0
#     cons_ign = 0

#     # filtering all start events for the day
#     event_5 = temp_df_event[(temp_df_event['type'] == 5)]
#     event_1 = temp_df_event[(temp_df_event['type'] == 1)]
    
#     try:
#       initial_vol = temp_df_fuel.loc[0, 'rV']
#       final_vol = temp_df_fuel.loc[len(temp_df_fuel)-1, 'rV']
#     except:
#       initial_vol =  -1
#       final_vol = -1

#     # Iterating over filtered events to calculate dg ignition time and ignition based dg Wh
#     for evi, evrow in event_5.iterrows():
#       try:
#         dq = datetime.fromtimestamp(evrow.ignition_list[1])
#         dq_ist = dq.astimezone(IST)
#         if dq_ist.date() == d:
#           dg_ign_time += evrow.ignition_list[1] - evrow.ignition_list[0]
#         else:
#           q = datetime.combine(d, time(18, 30, 0))
#           q = (q - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
#           dg_ign_time += q - evrow.ignition_list[0]

#         dg_wh_df = temp_df[(temp_df['unixStamps'] <= evrow.ignition_list[1]) & (temp_df['unixStamps'] >= evrow.ignition_list[0])]


#         x_dg = dg_wh_df['unixStampsDiff'].values
#         y_dg = dg_wh_df[f'acwatts'].values

#         dgwh = np.trapz(y_dg, x_dg)

#         dg_wh += dgwh

#         try:
#           cons_df = temp_df_fuel[(temp_df_fuel['unixStamps'] <= (evrow.ignition_list[1]+60)) & (temp_df_fuel['unixStamps'] >= (evrow.ignition_list[0]-60))].reset_index(drop=True)
#         except Exception as e:
#           print(f"{e} {evrow.ignition_list}")
#           cons_df = temp_df_fuel[temp_df_fuel['unixStamps'] >= (evrow.ignition_list[0]-60)].reset_index(drop=True)

#         if len(cons_df) > 0:
#           cons_ign += cons_df.loc[0, 'rV'] - cons_df.loc[len(cons_df)-1, 'rV']

#       except Exception as e:
#         print(f"event 5: {e} {evrow.ignition_list}")
    
#     # Iterating over filtered events to calculate refuelling
#     for evi, evrow in event_1.iterrows():
#       try:
#         try:
#           refuel_df = temp_df_fuel[(temp_df_fuel['unixStamps'] <= evrow.refuel_list[1]) & (temp_df_fuel['unixStamps'] >= evrow.refuel_list[0])].reset_index(drop=True)
#         except Exception as e:
#           print(f"{e} {evrow.refuel_list}")
#           refuel_df = temp_df_fuel[temp_df_fuel['unixStamps'] >= evrow.refuel_list[0]].reset_index(drop=True)

#         if len(refuel_df) > 0:
#           refuel += refuel_df.loc[len(refuel_df)-1, 'rV'] - refuel_df.loc[0, 'rV']


#       except Exception as e:
#         print(f"event 1: {e}")

#     try:
#       _, e_df2 = [x for _, x in temp_df.groupby(temp_df[f'ebwatts'] > 100)]
#       e_df2_list = [d for _, d in e_df2.groupby(e_df2.index - np.arange(len(e_df2)))]
#     except:
#       print(f"EB issue: {d}")
#       e_df2_list = []

#     try:
#       _, a_df2 = [x for _, x in temp_df.groupby(temp_df[f'acwatts'] < 2)]
#       a_df2_list = [d for _, d in a_df2.groupby(a_df2.index - np.arange(len(a_df2)))]
#     except:
#       print(f"AC issue: {d}")
#       a_df2_list = []

#     try:
#       _, ac_df2 = [x for _, x in temp_df.groupby(temp_df[f'acwatts'] < 100)]
#       ac_df2_list = [d for _, d in ac_df2.groupby(ac_df2.index - np.arange(len(ac_df2)))]
#     except:
#       print(f"AC issue: {d}")
#       ac_df2_list = []

#     try:
#       _, d_df2 = [x for _, x in temp_df.groupby(temp_df[f'dgwatts'] > 100)]
#       d_df2_list = [d for _, d in d_df2.groupby(d_df2.index - np.arange(len(d_df2)))]
#     except:
#       print(f"DG issue: {d}")
#       d_df2_list = []

#     try:
#       _, c_df2 = [x for _, x in temp_df.groupby(temp_df[f'ibattTOT'] > 20)]
#       c_df2_list = [d for _, d in c_df2.groupby(c_df2.index - np.arange(len(c_df2)))]
#     except:
#       print(f"Charge issue: {d}")
#       c_df2_list = []

#     try:
#       _, di_df2 = [x for _, x in temp_df.groupby(temp_df[f'ibattTOT'] < -20)]
#       di_df2_list = [d for _, d in di_df2.groupby(di_df2.index - np.arange(len(di_df2)))]
#     except:
#       print(f"Discharge issue: {d}")
#       di_df2_list = []

# # ################################################
#     try:
#       _, ch_df2 = [x for _, x in temp_df.groupby((temp_df[f'dcv'] < 51) & (temp_df[f'acwatts'] > 3000))]
#       ch_df2_list = [d for _, d in ch_df2.groupby(ch_df2.index - np.arange(len(ch_df2)))]
#     except:
#       print(f"Charge issue: {d}")
#       ch_df2_list = []

#     try:
#       _, dis_df2 = [x for _, x in temp_df.groupby((temp_df[f'dcv'] < 51) & (temp_df[f'acwatts'] < 4000))]
#       dis_df2_list = [d for _, d in dis_df2.groupby(dis_df2.index - np.arange(len(dis_df2)))]
#     except:
#       print(f"Discharge issue: {d}")
#       dis_df2_list = []

# ##################################################

#     for e_df in e_df2_list:
#       eb_time += e_df['unixStamps'].iloc[-1] - e_df['unixStamps'].iloc[0]

#     for a_df in a_df2_list:
#       ac_zero_time += a_df['unixStamps'].iloc[-1] - a_df['unixStamps'].iloc[0]

#     for d_df in d_df2_list:
#       dg_time += d_df['unixStamps'].iloc[-1] - d_df['unixStamps'].iloc[0]

#     for c_df in c_df2_list:
#       bat_char_time += c_df['unixStamps'].iloc[-1] - c_df['unixStamps'].iloc[0]

#     for di_df in di_df2_list:
#       bat_dischar_time += di_df['unixStamps'].iloc[-1] - di_df['unixStamps'].iloc[0]

#     ########################################################

#     for ch_df in ch_df2_list:
#       bat_char_time2 += ch_df['unixStamps'].iloc[-1] - ch_df['unixStamps'].iloc[0]

#     for dis_df in dis_df2_list:
#       bat_dischar_time2 += dis_df['unixStamps'].iloc[-1] - dis_df['unixStamps'].iloc[0]

#     #########################################################

#     for ac_df in ac_df2_list:
#       ac_100_time += ac_df['unixStamps'].iloc[-1] - ac_df['unixStamps'].iloc[0]

#     projected_dg_hrs = (dg_ign_time - dg_time)/3600 if (((dg_ign_time - dg_time)/3600 > 0.2) and (missing_hrs > 0)) else 0
#     projected_eb_hrs = (missing_hrs - projected_dg_hrs) if missing_hrs > 0 else 0
#     projected_dg_wh = projected_dg_hrs*median_wh
#     projected_eb_wh = projected_eb_hrs*median_wh

#     output_df[STRUCT[6]].append(projected_eb_hrs)
#     output_df[STRUCT[7]].append(projected_eb_wh)
#     output_df[STRUCT[14]].append(projected_dg_hrs)
#     output_df[STRUCT[15]].append(projected_dg_wh)

#     if missing_hrs == 0:
#       output_df[STRUCT[2]].append(eb_time/3600)
#     else:
#       output_df[STRUCT[2]].append(23.99-missing_hrs-(dg_time/3600)-(ac_100_time/3600))
#     output_df[STRUCT[9]].append(dg_time/3600)
#     output_df[STRUCT[10]].append(dg_ign_time/3600)
#     output_df[STRUCT[17]].append(bat_char_time/3600)
#     output_df[STRUCT[19]].append(bat_dischar_time/3600)
#     output_df[STRUCT[21]].append(ac_zero_time/3600)

#     output_df[STRUCT[18]].append(bat_char_time2/3600)
#     output_df[STRUCT[20]].append(bat_dischar_time2/3600)
#     output_df[STRUCT[22]].append(ac_100_time/3600)

#     output_df[STRUCT[11]].append((dg_time-dg_ign_time)/3600)

#     output_df[STRUCT[13]].append(dg_wh)

#     if site_code == "dand":
#       if boundary_flag:
#         print(f"Boundary: {d}")
#         eb_wh = twh - dg_wh
#       else:
#         eb_wh = twh - dg_wh - projected_dg_wh - projected_eb_wh
#     else:
#       if boundary_flag:
#         print(f"Boundary: {d}")
#         eb_wh = ebwh
#       else:
#         eb_wh = twh - dg_wh - projected_dg_wh - projected_eb_wh

#     output_df[STRUCT[5]].append(eb_wh)

#     output_df[STRUCT[8]].append(eb_wh + projected_eb_wh)
#     output_df[STRUCT[16]].append(dg_wh + projected_dg_wh)

#     if missing_hrs == 0:
#       if site_code == 'dand':
#         output_df[STRUCT[24]].append(dg_wh + projected_dg_wh + projected_eb_wh + eb_wh)
#       else:
#         output_df[STRUCT[24]].append(twh)
#     else:
#       output_df[STRUCT[24]].append(dg_wh + projected_dg_wh + projected_eb_wh + eb_wh)


#     consumption = initial_vol - final_vol + refuel

#     # if refuel > 1:
#     #   cons_final = cons_ign
#     #   output_df[STRUCT[27]].append(cons_ign)
#     #   output_df[STRUCT[28]].append(consumption-cons_ign)
#     # else:
#     cons_final = consumption
#     output_df[STRUCT[27]].append(consumption)
#     output_df[STRUCT[28]].append(0)

    

#     if site_code == 'dand':
#       try:
#         cph= cons_final/(dg_ign_time/3600)
#       except:
#         cph = 0
#       idling_time = 0
#     else:
#       if (dg_time-dg_ign_time)>0:
#         idling_time = 0
#         cph= cons_final/((dg_time-idling_time)/3600)
#       else:
#         idling_time = (dg_ign_time - dg_time - projected_dg_hrs*3600)
#         cph= cons_final/((dg_ign_time-idling_time)/3600)

#     output_df[STRUCT[25]].append(initial_vol)
#     output_df[STRUCT[26]].append(final_vol)
    
#     output_df[STRUCT[29]].append(refuel)
#     output_df[STRUCT[30]].append(idling_time/3600)
#     output_df[STRUCT[31]].append(cph)

#   return pd.DataFrame(output_df)
