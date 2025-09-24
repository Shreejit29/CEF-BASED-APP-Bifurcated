import pandas as pd
import numpy as np
from utils import time_to_decimal_hours

def load_and_clean(excel_file, remove_first=True):
    df = pd.read_excel(excel_file, sheet_name=0)
    df['Time_Hours'] = df['Time'].apply(time_to_decimal_hours)
    df = df.drop(['Time','Date'], axis=1)
    df = df[df['Current (mA)'] != 0].reset_index(drop=True)

    # Detect phase ends
    df['Sign'] = df['Current (mA)'] > 0
    df['Change'] = df['Sign'] != df['Sign'].shift()
    ends = [i-1 for i in df.index if df.loc[i,'Change']]
    if df.iloc[-1]['Current (mA)'] < 0: ends.append(len(df)-1)

    df_cycles = df.iloc[ends].reset_index(drop=True)
    df_cycles.insert(1,'Cycle_Number',range(1,len(df_cycles)+1))
    return compute_features(df_cycles, remove_first)

def compute_features(df, remove_conditioning):
    df['Charge_Capacity']    = df.apply(lambda r: r['Capacity (mAh)'] if r['Current (mA)']>0 else 0, axis=1)
    df['Discharge_Capacity'] = df['Charge_Capacity'].shift(-1).fillna(0)
    df['Charge_Energy']      = df.apply(lambda r: r['Energy (mWh)'] if r['Current (mA)']>0 else 0, axis=1)
    df['Discharge_Energy']   = df['Charge_Energy'].shift(-1).fillna(0)

    mask = (df['Charge_Capacity']>0)&(df['Discharge_Capacity']>0)&\
           (df['Charge_Energy']>0)&(df['Discharge_Energy']>0)
    df = df[mask].reset_index(drop=True)
    df['Cycle_Number'] = range(1,len(df)+1)

    CE = df['Discharge_Capacity']/df['Charge_Capacity']
    EE = df['Discharge_Energy']/df['Charge_Energy']
    df['Coulombic_Efficiency'], df['Energy_Efficiency'] = CE, EE
    df['CEF'] = 2/(1/np.exp(-10*(1-CE)) + 1/np.exp(-10*(1-EE)))

    if remove_conditioning:
        df = df.iloc[1:].reset_index(drop=True)
        df['Cycle_Number'] = range(1,len(df)+1)
    return df
