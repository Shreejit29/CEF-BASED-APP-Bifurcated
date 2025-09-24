import pandas as pd
import numpy as np
from utils import time_to_decimal_hours

REQUIRED_COLS = ['Sr. No.', 'Time', 'Date', 'Voltage (mV)', 'Current (mA)', 'Capacity (mAh)', 'Energy (mWh)']

def load_excel_first_sheet(uploaded_file):
    excel = pd.ExcelFile(uploaded_file)
    sheet = excel.sheet_names[0]
    df = pd.read_excel(uploaded_file, sheet_name=sheet)
    return sheet, df

def build_final_dataset(df_raw: pd.DataFrame, remove_first_row: bool) -> pd.DataFrame:
    # Validate required columns exist
    missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df_raw.copy()
    df['Time_Hours'] = df['Time'].apply(time_to_decimal_hours)
    df_cleaned = df.drop(['Time', 'Date'], axis=1)

    columns_order = ['Sr. No.', 'Time_Hours', 'Voltage (mV)', 'Current (mA)', 'Capacity (mAh)', 'Energy (mWh)']
    df_final = df_cleaned[columns_order]
    df_final = df_final[df_final['Current (mA)'] != 0].reset_index(drop=True)

    # Phase transitions
    df_final['Current_Sign'] = df_final['Current (mA)'] > 0
    df_final['Sign_Change'] = df_final['Current_Sign'] != df_final['Current_Sign'].shift(1)

    end_of_phases = []
    for i in range(1, len(df_final)):
        if bool(df_final.iloc[i]['Sign_Change']):
            end_of_phases.append(i-1)
    if len(df_final) and df_final.iloc[-1]['Current (mA)'] < 0:
        end_of_phases.append(len(df_final)-1)

    if len(end_of_phases) == 0:
        # No complete phases found; return empty structured frame
        empty = df_final.head(0).drop(columns=['Current_Sign', 'Sign_Change'])
        empty['Charge_Capacity'] = []
        empty['Discharge_Capacity'] = []
        empty['Charge_Energy'] = []
        empty['Discharge_Energy'] = []
        empty['Coulombic_Efficiency'] = []
        empty['Energy_Efficiency'] = []
        empty['CEF'] = []
        empty.insert(1, 'Cycle_Number', [])
        return empty

    final_dataset = df_final.iloc[end_of_phases].copy().reset_index(drop=True)
    final_dataset['Sr. No.'] = range(1, len(final_dataset)+1)
    final_dataset = final_dataset.drop(['Current_Sign', 'Sign_Change'], axis=1)

    # Capacities/Energy by phase
    final_dataset['Charge_Capacity'] = final_dataset.apply(
        lambda row: row['Capacity (mAh)'] if row['Current (mA)'] > 0 else 0, axis=1
    )
    final_dataset['Discharge_Capacity'] = final_dataset.apply(
        lambda row: row['Capacity (mAh)'] if row['Current (mA)'] < 0 else 0, axis=1
    )
    final_dataset['Charge_Energy'] = final_dataset.apply(
        lambda row: row['Energy (mWh)'] if row['Current (mA)'] > 0 else 0, axis=1
    )
    final_dataset['Discharge_Energy'] = final_dataset.apply(
        lambda row: row['Energy (mWh)'] if row['Current (mA)'] < 0 else 0, axis=1
    )

    # Align discharge to following row
    final_dataset['Discharge_Capacity'] = final_dataset['Discharge_Capacity'].shift(-1).fillna(0)
    final_dataset['Discharge_Energy'] = final_dataset['Discharge_Energy'].shift(-1).fillna(0)

    # Keep only valid paired rows
    mask = (final_dataset['Charge_Capacity'] > 0) & (final_dataset['Discharge_Capacity'] > 0) & \
           (final_dataset['Charge_Energy'] > 0) & (final_dataset['Discharge_Energy'] > 0)
    final_dataset = final_dataset[mask].reset_index(drop=True)
    final_dataset['Sr. No.'] = range(1, len(final_dataset)+1)
    final_dataset.insert(1, 'Cycle_Number', range(1, len(final_dataset)+1))

    if len(final_dataset) == 0:
        return final_dataset

    # Efficiencies and CEF
    final_dataset['Coulombic_Efficiency'] = final_dataset['Discharge_Capacity'] / final_dataset['Charge_Capacity']
    final_dataset['Energy_Efficiency'] = final_dataset['Discharge_Energy'] / final_dataset['Charge_Energy']

    CE = final_dataset['Coulombic_Efficiency']
    EE = final_dataset['Energy_Efficiency']
    final_dataset['CEF'] = 2 / (1 / np.exp(-10 * (1 - CE)) + 1 / np.exp(-10 * (1 - EE)))

    if remove_first_row and len(final_dataset) >= 1:
        final_dataset = final_dataset.iloc[1:].reset_index(drop=True)
        final_dataset['Sr. No.'] = range(1, len(final_dataset)+1)
        final_dataset['Cycle_Number'] = range(1, len(final_dataset)+1)

    return final_dataset
