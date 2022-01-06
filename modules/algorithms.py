import pandas as pd
import numpy as np

def detect_dg_idling(df_event, df_elm, df_fuel, dl):
    for i, row in df_event.iterrows():
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
                except:
                    pass
                if len(idl)>1 and (idl.loc[len(idl)-1, 'ts'] - idl.loc[0, 'ts']) > dl:
                    print(f"\n DG Idling from: {idl.sort_values(by = 'ts').ts.values[0]} to: {idl.sort_values(by = 'ts').ts.values[-1]}")
                    print(idl)

def detect_fuel_drain():
    pass