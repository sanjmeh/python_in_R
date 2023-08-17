import pandas as pd 
# import numpy as np
# import matplotlib.pyplot as plt
from datetime import datetime
# from haversine import haversine, Unit
from tqdm import tqdm
import pytz
# import pyreadr
# import math
# import os
tqdm.pandas()
# os.getcwd()
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 20)

cst = pd.read_csv("~/JSW-VTPL/python/data/cst_all_copy.csv")
allmods = pd.read_csv("~/JSW-VTPL/python/data/allmods.csv")
cst['ts'] = pd.to_datetime(cst['ts'])
# print(cst.head(5))
# print(allmods.shape)

def timestamp_(date):
    formatted_datetime = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
#     print('T1')
    return formatted_datetime

def Utc_to_Ist(utc_time1):

    utc_time = datetime.strptime(str(utc_time1), "%Y-%m-%d %H:%M:%S")
    ist_timezone = pytz.timezone("Asia/Kolkata")
    ist_time = utc_time.replace(tzinfo=pytz.UTC).astimezone(ist_timezone).strftime("%Y-%m-%d %H:%M:%S")

    return ist_time

term_df = cst.query("termid==1204000784")
try:
    term_df['ts'] = term_df['ts'].progress_apply(timestamp_)
except:
    term_df['ts'] = term_df['ts'].progress_apply(Utc_to_Ist)

term_df.to_csv("~/JSW-VTPL/python/data/one_termid_data.csv")