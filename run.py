from argparse import ArgumentParser
from datetime import date

from modules.data import get_df_from_rds

def main():
    parser = ArgumentParser()

    parser.add_argument("--idling", help="To run idling algorithm", action="store_true")
    parser.add_argument("--drain", help="To run drain algorithm", action="store_true")

    # parser.add_argument("-s", "--sites", nargs='+', help="The name of sites: {bagru1, bagru2, dand, jobner, sawarda}", required=True)
    parser.add_argument("--start", type=date.fromisoformat, help="The starting date for data")
    parser.add_argument("--end", type=date.fromisoformat, help="The ending date for data")

    parser.add_argument("--file")

    args = parser.parse_args()

    df = get_df_from_rds('sample_data/' + args.file)
    df.to_csv('sample.csv', index=False)


if __name__=='__main__':
    main()