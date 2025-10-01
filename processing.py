import pandas as pd
import numpy as np
from io import BytesIO
from utils import time_to_decimal_hours, canonicalize_columns

# Canonical raw schema as used by build_final_dataset
REQUIRED_COLS = ['Sr. No.', 'Time', 'Date', 'Voltage (mV)', 'Current (mA)', 'Capacity (mAh)', 'Energy (mWh)']

# Flexible alias map so differently labeled inputs are auto-canonicalized
ALIAS_MAP = {
    "Cycle_Number": ["Cycle_Number", "Cycle", "Cycle No", "cycle number", "Cycle_Index", "Cycle_index", "Cycle Index"],
    "Time": ["Time", "time stamp", "t", "Duration"],
    "Date": ["Date", "Datetime", "Timestamp", "date time"],
    "Voltage (mV)": ["Voltage (mV)", "Voltage_mV", "VmV", "Voltage", "Voltage (V)"],
    "Current (mA)": ["Current (mA)", "Current_mA", "ImA", "Current", "Current (A)"],
    "Capacity (mAh)": ["Capacity (mAh)", "Capacity_mAh", "QmAh", "Capacity", "Cap (mAh)", "Capacity (Ah)", "Capacity(Ah)"],
    "Energy (mWh)": ["Energy (mWh)", "Energy_mWh", "EmWh", "Energy", "Energy (Wh)", "Energy(Wh)"],
    "Charge_Capacity": ["Charge Capacity", "Qchg", "Q_charge", "Qcharge", "Charge_Capacity"],
    "Discharge_Capacity": ["Discharge Capacity", "Qdchg", "Q_discharge", "Qdischarge","Discharge_Capacity"],
    "Charge_Energy": ["Charge Energy", "Echg", "E_charge", "Echarge", "Charge_Energy"],
    "Discharge_Energy": ["Discharge Energy", "Edchg", "E_discharge", "Edischarge", "Discharge_Energy"],
    "Coulombic_Efficiency": ["Coulombic efficiency","Coulombic Efficiency","CE","CoulombicEff","Coulombic_Eff","C Eff"],
    "Energy_Efficiency": ["Energy efficiency","Energy Efficiency","EE","EnergyEff","Energy_Eff"],
    "CEF": ["CEF","cef","CEF value","cef_value"]
}

def load_excel_first_sheet(uploaded_file):
    excel = pd.ExcelFile(uploaded_file)
    sheet = excel.sheet_names[0]
    df = pd.read_excel(uploaded_file, sheet_name=sheet)
    # Auto-canonicalize raw columns for convenience
    df, _ = canonicalize_columns(df, ALIAS_MAP)
    return sheet, df

def build_final_dataset(df_raw: pd.DataFrame, remove_first_row: bool) -> pd.DataFrame:
    # Canonicalize first in case user provided variants
    df_raw, _ = canonicalize_columns(df_raw, ALIAS_MAP)

    missing = [c for c in ['Time','Date','Current (mA)','Capacity (mAh)','Energy (mWh)'] if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required raw fields after mapping/canonicalization: {missing}")

    df = df_raw.copy()

    # Create Sr. No. if missing to stabilize ordering
    if 'Sr. No.' not in df.columns:
        df.insert(0, 'Sr. No.', range(1, len(df)+1))

    df['Time_Hours'] = df['Time'].apply(time_to_decimal_hours)
    df_cleaned = df.drop([c for c in ['Time', 'Date'] if c in df.columns], axis=1)

    columns_order = ['Sr. No.', 'Time_Hours', 'Voltage (mV)', 'Current (mA)', 'Capacity (mAh)', 'Energy (mWh)']
    columns_order = [c for c in columns_order if c in df_cleaned.columns]
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
        empty = df_final.head(0).drop(columns=[c for c in ['Current_Sign','Sign_Change'] if c in df_final.columns], errors='ignore')
        for c in ["Charge_Capacity","Discharge_Capacity","Charge_Energy","Discharge_Energy",
                  "Coulombic_Efficiency","Energy_Efficiency","CEF"]:
            empty[c] = []
        empty.insert(0, 'Cycle_Number', [])
        return empty

    final_dataset = df_final.iloc[end_of_phases].copy().reset_index(drop=True)
    final_dataset['Sr. No.'] = range(1, len(final_dataset)+1)
    final_dataset = final_dataset.drop(columns=[c for c in ['Current_Sign','Sign_Change'] if c in final_dataset.columns], errors='ignore')

    # Capacities/Energy by phase
    final_dataset['Charge_Capacity'] = final_dataset.apply(
        lambda row: row['Capacity (mAh)'] if row['Current (mA)'] > 0 else 0, axis=1
    ) if 'Capacity (mAh)' in final_dataset.columns else 0
    final_dataset['Discharge_Capacity'] = final_dataset.apply(
        lambda row: row['Capacity (mAh)'] if row['Current (mA)'] < 0 else 0, axis=1
    ) if 'Capacity (mAh)' in final_dataset.columns else 0
    final_dataset['Charge_Energy'] = final_dataset.apply(
        lambda row: row['Energy (mWh)'] if row['Current (mA)'] > 0 else 0, axis=1
    ) if 'Energy (mWh)' in final_dataset.columns else 0
    final_dataset['Discharge_Energy'] = final_dataset.apply(
        lambda row: row['Energy (mWh)'] if row['Current (mA)'] < 0 else 0, axis=1
    ) if 'Energy (mWh)' in final_dataset.columns else 0

    # Align discharge to following row
    final_dataset['Discharge_Capacity'] = pd.Series(final_dataset['Discharge_Capacity']).shift(-1).fillna(0)
    final_dataset['Discharge_Energy'] = pd.Series(final_dataset['Discharge_Energy']).shift(-1).fillna(0)

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
    ["CEF"],
    ["Coulombic_Efficiency", "Energy_Efficiency"],
    ["Discharge_Capacity", "Charge_Capacity", "Discharge_Energy", "Charge_Energy"]
]
OPTIONAL_COMMON = ["Cycle_Number", "Charge_Capacity", "Discharge_Capacity",
                   "Charge_Energy", "Discharge_Energy",
                   "Coulombic_Efficiency", "Energy_Efficiency", "CEF"]

def _read_any(uploaded_file):
    name = getattr(uploaded_file, "name", "")
    if name and name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        fmt = "csv"
    else:
        excel = pd.ExcelFile(uploaded_file)
        sheet = excel.sheet_names[0]
        df = pd.read_excel(uploaded_file, sheet_name=sheet)
        fmt = "excel"
    # Canonicalize for flexibility
    df, _ = canonicalize_columns(df, ALIAS_MAP)
    return df, fmt

def validate_processed_dataset(uploaded_file):
    notes = []
    df, fmt = _read_any(uploaded_file)
    if df.empty:
        raise ValueError("Processed dataset is empty.")
    return validate_processed_dataframe(df)

def validate_processed_dataframe(df_in: pd.DataFrame):
    df = df_in.copy()
    notes = []

    # Ensure Cycle_Number exists
    if "Cycle_Number" not in df.columns:
        df = df.reset_index(drop=True)
        df.insert(0, "Cycle_Number", range(1, len(df) + 1))
        notes.append("Cycle_Number was missing and has been created as a simple sequence.")

    # Compute efficiencies if possible
    if "Coulombic_Efficiency" not in df.columns and {"Discharge_Capacity", "Charge_Capacity"}.issubset(df.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            df["Coulombic_Efficiency"] = np.where(df["Charge_Capacity"] > 0,
                                                  df["Discharge_Capacity"] / df["Charge_Capacity"], np.nan)
        notes.append("Coulombic_Efficiency computed from capacities.")
    if "Energy_Efficiency" not in df.columns and {"Discharge_Energy", "Charge_Energy"}.issubset(df.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            df["Energy_Efficiency"] = np.where(df["Charge_Energy"] > 0,
                                               df["Discharge_Energy"] / df["Charge_Energy"], np.nan)
        notes.append("Energy_Efficiency computed from energies.")

    # Compute CEF if possible
    if "CEF" not in df.columns and {"Coulombic_Efficiency", "Energy_Efficiency"}.issubset(df.columns):
        CE = df["Coulombic_Efficiency"].astype(float)
        EE = df["Energy_Efficiency"].astype(float)
        df["CEF"] = 2 / (1 / np.exp(-10 * (1 - CE)) + 1 / np.exp(-10 * (1 - EE)))
        notes.append("CEF computed from CE and EE.")

    # Validate coverage
    has_ok = ("CEF" in df.columns) or ({"Coulombic_Efficiency","Energy_Efficiency"}.issubset(df.columns)) \
             or ({"Discharge_Capacity","Charge_Capacity","Discharge_Energy","Charge_Energy"}.issubset(df.columns))
    if not has_ok:
        raise ValueError("Processed dataset lacks required columns. Provide CEF or CE+EE or full capacity/energy pairs.")

    # Coerce numeric types for downstream stats
    for c in ["Charge_Capacity","Discharge_Capacity","Charge_Energy","Discharge_Energy","Coulombic_Efficiency","Energy_Efficiency","CEF"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    preferred_order = ["Cycle_Number",
                       "Charge_Capacity", "Discharge_Capacity",
                       "Charge_Energy", "Discharge_Energy",
                       "Coulombic_Efficiency", "Energy_Efficiency",
                       "CEF"]
    final_cols = [c for c in preferred_order if c in df.columns]
    final_dataset = df[final_cols].copy()
    return final_dataset, notes
