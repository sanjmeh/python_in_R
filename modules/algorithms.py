import pandas as pd

def detect_dg_idling(df_event, df_elm, df_fuel):
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

            if len(idl_df)>1:
                print(f"\n DG Idling from: {idl_df.sort_values(by = 'ts').ts.values[0]} to: {idl_df.sort_values(by = 'ts').ts.values[-1]}")
                print(idl_df)

def detect_fuel_drain():
    pass