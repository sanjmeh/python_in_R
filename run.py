import pandas as pd

from argparse import ArgumentParser
from datetime import date

from tqdm.std import tqdm
from modules.config import MONTHS, TOPICS, VEHICLE_DIR, VEHICLE_LIST

from modules.data import filter_by_topic, get_df_from_rds, get_data_range, filter_by_topic_param, localize_df
from modules.config import MONTHS, BASE_DIR
from modules.algorithms import detect_dg_idling, detect_fuel_drain, detect_fuel_drain_ind

def main():
    parser = ArgumentParser()

    parser.add_argument("--idling", help="To run idling algorithm", action="store_true")
    parser.add_argument("--drain", help="To run drain algorithm", action="store_true")
    parser.add_argument("--drain-ind", help="To run drain independent algorithm", action="store_true")

    parser.add_argument("-s", "--sites", nargs='+', help="The name of sites: {bagru1, bagru2, dand, jobner, sawarda}", default=VEHICLE_LIST)
    # parser.add_argument("-t", "--topics", nargs='+', help="The topic strings", required=True)
    parser.add_argument("--start", type=str, help="The starting date for data")
    parser.add_argument("--end", type=str, help="The ending date for data")

    parser.add_argument("--delta_t", type=int, help="Threshold in minutes for idling detection", default=10)

    parser.add_argument("--ist", type=int, help="Threshold in minutes for idling detection", default=10)
    parser.add_argument("--ie", type=int, help="Threshold in minutes for idling detection", default=10)

    args = parser.parse_args()

    base_dir = BASE_DIR
    fuel_path = base_dir + f'fuellvl/{args.sites[0]}_fdt.RDS'
    event_path = base_dir + f'dgevents/{args.sites[0]}_eventdt.RDS'
    m_int = int(args.start.split('-')[1])
    elm_path = base_dir + f'elm/{args.sites[0]}/{MONTHS[m_int]}_data.RDS'

    print(f"Building Dataframes from {event_path, fuel_path} ...")

    dl = pd.Timedelta(f"{args.delta_t} minutes")

    if args.drain_ind:
        fuel_df = pd.read_csv(VEHICLE_DIR+'vehicle_data.csv')
        fuel_df = localize_df(fuel_df)
    else:
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

    if args.drain_ind:
        df_final_list = []
        print("Starting drain detection...")
        for i, vehicle in tqdm(enumerate(args.sites[args.ist: args.ie])):
            fuel_df_temp = filter_by_topic(fuel_df, vehicle)
            print(fuel_df_temp)
            df_final_list.append(detect_fuel_drain_ind(fuel_df_temp, name=vehicle))
            print(f"vehicle {i+1}: {vehicle} done processing.")
        print("Detection done")

        print("Saving...")
        pd.concat(df_final_list).reset_index(drop=True).to_csv('slope_fuel_report.csv', index=False)
        print("Saved.")




if __name__=='__main__':
    main()