import pandas as pd

from argparse import ArgumentParser
from datetime import date
from modules.config import MONTHS, TOPICS

from modules.data import get_df_from_rds, get_data_range, filter_by_topic_param
from modules.config import MONTHS, BASE_DIR
from modules.algorithms import detect_dg_idling, detect_fuel_drain

def main():
    parser = ArgumentParser()

    parser.add_argument("--idling", help="To run idling algorithm", action="store_true")
    parser.add_argument("--drain", help="To run drain algorithm", action="store_true")

    parser.add_argument("-s", "--sites", nargs='+', help="The name of sites: {bagru1, bagru2, dand, jobner, sawarda}", required=True)
    # parser.add_argument("-t", "--topics", nargs='+', help="The topic strings", required=True)
    parser.add_argument("--start", type=str, help="The starting date for data")
    parser.add_argument("--end", type=str, help="The ending date for data")

    parser.add_argument("--delta_t", type=int, help="Threshold in minutes for idling detection", default=10)

    args = parser.parse_args()

    base_dir = BASE_DIR
    fuel_path = base_dir + f'fuellvl/{args.sites[0]}_fdt.RDS'
    event_path = base_dir + f'dgevents/{args.sites[0]}_eventdt.RDS'
    m_int = int(args.start.split('-')[1])
    elm_path = base_dir + f'elm/{args.sites[0]}/{MONTHS[m_int]}_data.RDS'

    print(f"Building Dataframes from {event_path, fuel_path} ...")

    dl = pd.Timedelta(f"{args.delta_t} minutes")
    
    fuel_df = get_data_range(get_df_from_rds(fuel_path), args.start, args.end)
    print(f"Fuel df fetched.")
    event_df = get_data_range(get_df_from_rds(event_path), args.start, args.end)
    print(f"Event df fetched.")

    if args.idling:
        print(f"Building Dataframe from {elm_path} with topics {TOPICS[args.sites[0]]} ...")
        elm_df = filter_by_topic_param(get_data_range(get_df_from_rds(elm_path), args.start, args.end), TOPICS[args.sites[0]], 155)
        print(f"Elm df fetched.")

        print("Starting idling detection...")
        detect_dg_idling(event_df, elm_df, fuel_df, dl)
        print("Detection done")

    if args.drain:
        print("Starting drain detection...")
        detect_fuel_drain(event_df, fuel_df)
        print("Detection done")




if __name__=='__main__':
    main()