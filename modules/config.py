import pytz
import pandas as pd

from pathlib import Path
import os

DIR = Path(__file__).resolve().parent.parent.parent

IST = pytz.timezone('Asia/Kolkata')
BASE_DIR = os.path.join(DIR, 'data/')
VEHICLE_DIR = os.path.join(BASE_DIR, 'VehicleReports/')

VEHICLE_LIST = pd.read_csv(os.path.join(DIR, 'Mindshift-GH/vehicle_list.csv'), header=None).values

# print(type(VEHICLE_LIST))

MONTHS = {
    1: 'jan',
    2: 'feb',
    3: 'mar',
    4: 'apr',
    5: 'may',
    6: 'jun',
    7: 'jul',
    8: 'aug',
    9: 'sep',
    10: 'oct',
    11: 'nov',
    12: 'dec',
}

TOPICS = {
    'bagru1': ['bagru1/ELM8420', 'bagru1/ELM8420B', 'bagru1/ELM8420C'],
    'bagru2': ['bagru2/ELM8420', 'bagru2/ELM8420B', 'bagru2/ELM8420C'],
    'sawarda': ['saw001/ELM8420'],
    'jobner': ['jobner001/ELM8420']
}

SITE_CODES =  {
    'bagru1': 'bagru1',
    'bagru2': 'bagru2',
    'sawarda': 'saw001',
    'jobner': 'jobner001',
    'dand': 'dand',
}