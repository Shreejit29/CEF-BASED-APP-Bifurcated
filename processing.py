import pandas as pd
import numpy as np
from io import BytesIO
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
PROCESSED_REQUIRED_ANY = [
    # at least one of these groups must be present to proceed
    ["CEF"],
    ["Coulombic_Efficiency", "Energy_Efficiency"],
    ["Discharge_Capacity", "Charge_Capacity", "Discharge_Energy", "Charge_Energy"]
]
OPTIONAL_COMMON = ["Cycle_Number", "Charge_Capacity", "Discharge_Capacity",
                   "Charge_Energy", "Discharge_Energy",
                   "Coulombic_Efficiency", "Energy_Efficiency", "CEF"]

def _read_any(uploaded_file):
    name = getattr(uploaded_file, "name", "")
    if name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        fmt = "csv"
    else:
        # Excel
        excel = pd.ExcelFile(uploaded_file)
        sheet = excel.sheet_names[0]
        df = pd.read_excel(uploaded_file, sheet_name=sheet)
        fmt = "excel"
    return df, fmt

def validate_processed_dataset(uploaded_file):
    """
    Load a user-supplied processed dataset and ensure it is suitable for analysis.
    Returns (final_dataset, notes) where notes is a list of strings about inferred/recomputed fields.
    """
    notes = []
    df, fmt = _read_any(uploaded_file)
    if df.empty:
        raise ValueError("Processed dataset is empty.")

    cols = set(df.columns)

    # Ensure Cycle_Number exists or create a simple 1..N
    if "Cycle_Number" not in cols:
        df = df.reset_index(drop=True)
        df.insert(0, "Cycle_Number", range(1, len(df) + 1))
        notes.append("Cycle_Number was missing and has been created as a simple sequence.")

    # Try to compute efficiencies if capacities/energies exist
    if "Coulombic_Efficiency" not in cols and {"Discharge_Capacity", "Charge_Capacity"}.issubset(cols):
        with np.errstate(divide='ignore', invalid='ignore'):
            df["Coulombic_Efficiency"] = np.where(df["Charge_Capacity"] > 0,
                                                  df["Discharge_Capacity"] / df["Charge_Capacity"], np.nan)
        notes.append("Coulombic_Efficiency computed from capacities.")

    if "Energy_Efficiency" not in cols and {"Discharge_Energy", "Charge_Energy"}.issubset(cols):
        with np.errstate(divide='ignore', invalid='ignore'):
            df["Energy_Efficiency"] = np.where(df["Charge_Energy"] > 0,
                                               df["Discharge_Energy"] / df["Charge_Energy"], np.nan)
        notes.append("Energy_Efficiency computed from energies.")

    # Compute CEF if missing and CE/EE exist
    if "CEF" not in cols and {"Coulombic_Efficiency", "Energy_Efficiency"}.issubset(df.columns):
        CE = df["Coulombic_Efficiency"].astype(float)
        EE = df["Energy_Efficiency"].astype(float)
        df["CEF"] = 2 / (1 / np.exp(-10 * (1 - CE)) + 1 / np.exp(-10 * (1 - EE)))
        notes.append("CEF computed from CE and EE.")

    # Validate that at least one acceptable group is present now
    cols_after = set(df.columns)
    ok = any(set(group).issubset(cols_after) for group in PROCESSED_REQUIRED_ANY)
    if not ok:
        raise ValueError("Processed dataset lacks required columns. Provide CEF or CE+EE or full capacity/energy pairs.")

    # Keep only relevant analysis columns if present
    preferred_order = ["Cycle_Number",
                       "Charge_Capacity", "Discharge_Capacity",
                       "Charge_Energy", "Discharge_Energy",
                       "Coulombic_Efficiency", "Energy_Efficiency",
                       "CEF"]
    final_cols = [c for c in preferred_order if c in df.columns]
    final_dataset = df[final_cols].copy()
    return final_dataset, notes
