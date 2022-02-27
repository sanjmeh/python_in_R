import pandas as pd
import pytz
from pathlib import Path
import os

BASE = Path(__file__).parent

IST = pytz.timezone("Asia/Kolkata")
UTC = pytz.utc

SITE_REGISTER = pd.read_csv('register_data.csv')
SITE_CODES = {
    "bagru1": "bagru1",
    "bagru2": "bagru2",
    "dand": "dand",
    "jobner": "jobner001",
    "sawarda": "saw001",
}

with open(os.path.join(BASE, "struct.txt")) as f:
    STRUCT = f.read().split("\n")

WEEK_DAYS = {
    0:  "Monday",
    1:  "Tuesday",
    2:  "Wednesday",
    3:  "Thursday",
    4:  "Friday",
    5:  "Saturday",
    6:  "Sunday"
}