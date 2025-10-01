import pandas as pd
import numpy as np
from io import BytesIO
from utils import time_to_decimal_hours, canonicalize_columns

# Canonical raw schema as used by build_final_dataset
REQUIRED_COLS = ['Sr. No.', 'Time', 'Date', 'Voltage (mV)', 'Current (mA)', 'Capacity (mAh)', 'Energy (mWh)']

# Flexible alias map so differently labeled inputs are auto-canonicalized
ALIAS_MAP = {
    # indices / ids
    "Cycle_Number": ["Cycle_Number", "Cycle", "Cycle No", "cycle number", "Cycle_Index", "Cycle_index", "Cycle Index"],
    # time/date
    "Time": ["Time", "time stamp", "t", "Duration", "Test Time", "Record Time"],
    "Date": ["Date", "Datetime", "Timestamp", "date time", "Date Time"],
    # voltage/current with unit normalization
    "Voltage (mV)": ["Voltage (mV)", "Voltage_mV", "VmV", "Voltage", "Voltage (V)", "U/V", "Volt"],
    "Current (mA)": ["Current (mA)", "Current_mA", "ImA", "Current", "Current (A)", "I/A", "Ampere"],
    # total capacity/energy (signed)
    "Capacity (mAh)": ["Capacity (mAh)", "Capacity_mAh", "QmAh", "Capacity", "Cap (mAh)", "Capacity (Ah)", "Capacity(Ah)", "Charge/Discharge Capacity"],
    "Energy (mWh)": ["Energy (mWh)", "Energy_mWh", "EmWh", "Energy", "Energy (Wh)", "Energy(Wh)", "Charge/Discharge Energy"],
    # separated pairs if present
    "Charge_Capacity": ["Charge Capacity", "Qchg", "Q_charge", "Qcharge", "Charge_Capacity"],
    "Discharge_Capacity": ["Discharge Capacity", "Qdchg", "Q_discharge", "Qdischarge","Discharge_Capacity"],
    "Charge_Energy": ["Charge Energy", "Echg", "E_charge", "Echarge", "Charge_Energy"],
    "Discharge_Energy": ["Discharge Energy", "Edchg", "E_discharge", "Edischarge", "Discharge_Energy"],
    # efficiencies / CEF
    "Coulombic_Efficiency": ["Coulombic efficiency","Coulombic Efficiency","CE","CoulombicEff","Coulombic_Eff","C Eff"],
    "Energy_Efficiency": ["Energy efficiency","Energy Efficiency","EE","EnergyEff","Energy_Eff"],
    "CEF": ["CEF","cef","CEF value","cef_value"]
}

def _normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    # If voltage in V, convert to mV
    if "Voltage (mV)" in df.columns:
        # Heuristic: if values are mostly < 10, assume it's V and convert
        s = pd.to_numeric(df["Voltage (mV)"], errors="coerce")
        if s.notna().mean() > 0 and s.dropna().abs().median() < 10:
            df["Voltage (mV)"] = s * 1000.0
    # If current in A, convert to mA
    if "Current (mA)" in df.columns:
        s = pd.to_numeric(df["Current (mA)"], errors="coerce")
        if s.notna().mean() > 0 and s.dropna().abs().median() < 5:  # many cyclers use < 5 A
            # If median < 5, it could still be mA; check max
            if s.dropna().abs().max() <= 20:  # likely Amps
                df["Current (mA)"] = s * 1000.0
    # If energy in Wh, convert to mWh
    if "Energy (mWh)" in df.columns:
        s = pd.to_numeric(df["Energy (mWh)"], errors="coerce")
        if s.notna().mean() > 0 and s.dropna().abs().median() < 10:  # small Wh typical
            # If values like 0.1..5, may be Wh; convert to mWh
            if s.dropna().abs().max() < 50:
                df["Energy (mWh)"] = s * 1000.0
    # If capacity in Ah, convert to mAh
    if "Capacity (mAh)" in df.columns:
        s = pd.to_numeric(df["Capacity (mAh)"], errors="coerce")
        if s.notna().mean() > 0 and s.dropna().abs().median() < 10:
            if s.dropna().abs().max() < 50:
                df["Capacity (mAh)"] = s * 1000.0
    return df

def load_excel_first_sheet(uploaded_file):
    excel = pd.ExcelFile(uploaded_file)
    sheet = excel.sheet_names[0]
    df = pd.read_excel(uploaded_file, sheet_name=sheet)
    df, _ = canonicalize_columns(df, ALIAS_MAP)
    df = _normalize_units(df)
    return sheet, df

def build_final_dataset(df_raw: pd.DataFrame, remove_first_row: bool) -> pd.DataFrame:
    # Canonicalize and normalize units first
    df_raw, _ = canonicalize_columns(df_raw, ALIAS_MAP)
    df_raw = _normalize_units(df_raw)

    # Ensure the minimal fields exist
    missing = [c for c in ['Time','Date','Current (mA)','Capacity (mAh)','Energy (mWh)'] if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required raw fields after mapping/canonicalization: {missing}")

    df = df_raw.copy()

    # Sr. No. if missing
    if 'Sr. No.' not in df.columns:
        df.insert(0, 'Sr. No.', range(1, len(df)+1))

    # Time to hours
    df['Time_Hours'] = df['Time'].apply(time_to_decimal_hours)

    # Trim and coerce numerics used later
    for c in ['Voltage (mV)','Current (mA)','Capacity (mAh)','Energy (mWh)']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop original time/date for compactness
    df_cleaned = df.drop([c for c in ['Time', 'Date'] if c in df.columns], axis=1)

    # Order columns for readability
    columns_order = ['Sr. No.', 'Time_Hours', 'Voltage (mV)', 'Current (mA)', 'Capacity (mAh)', 'Energy (mWh)']
    columns_order = [c for c in columns_order if c in df_cleaned.columns]
    df_final = df_cleaned[columns_order]

    # Remove pure rest rows (zero or near-zero current)
    df_final = df_final[~df_final['Current (mA)'].abs().lt(1e-6)].reset_index(drop=True)

    # Determine phase boundaries by sign changes
    sign = df_final['Current (mA)'] > 0
    sign_change = sign.ne(sign.shift(1))
    boundary_idx = [i-1 for i, ch in enumerate(sign_change) if ch and i > 0]

    # End discharge at file end if last segment is discharge
    if len(df_final) and (df_final.iloc[-1]['Current (mA)'] < 0):
        boundary_idx.append(len(df_final)-1)

    if len(boundary_idx) == 0:
        # Return structured empty with all expected cols
        empty = pd.DataFrame(columns=['Sr. No.','Time_Hours','Voltage (mV)','Current (mA)',
                                      'Capacity (mAh)','Energy (mWh)','Charge_Capacity','Discharge_Capacity',
                                      'Charge_Energy','Discharge_Energy','Coulombic_Efficiency','Energy_Efficiency','CEF','Cycle_Number'])
        return empty

    # Phase endpoints rows
    final_dataset = df_final.iloc[boundary_idx].copy().reset_index(drop=True)
    final_dataset['Sr. No.'] = range(1, len(final_dataset)+1)

    # Split signed totals into charge/discharge
    final_dataset['Charge_Capacity'] = np.where(final_dataset['Current (mA)'] > 0, final_dataset['Capacity (mAh)'], 0.0)
    final_dataset['Discharge_Capacity'] = np.where(final_dataset['Current (mA)'] < 0, final_dataset['Capacity (mAh)'], 0.0)
    final_dataset['Charge_Energy'] = np.where(final_dataset['Current (mA)'] > 0, final_dataset['Energy (mWh)'], 0.0)
    final_dataset['Discharge_Energy'] = np.where(final_dataset['Current (mA)'] < 0, final_dataset['Energy (mWh)'], 0.0)

    # Align discharge with next row (pairing with prior charge)
    final_dataset['Discharge_Capacity'] = final_dataset['Discharge_Capacity'].shift(-1).fillna(0.0)
    final_dataset['Discharge_Energy'] = final_dataset['Discharge_Energy'].shift(-1).fillna(0.0)

    # Keep rows where both charge and discharge are positive
    mask = (final_dataset['Charge_Capacity'] > 0) & (final_dataset['Discharge_Capacity'] > 0) & \
           (final_dataset['Charge_Energy'] > 0) & (final_dataset['Discharge_Energy'] > 0)
    final_dataset = final_dataset[mask].reset_index(drop=True)

    if final_dataset.empty:
        return final_dataset

    # Cycle number and re-index
    final_dataset.insert(1, 'Cycle_Number', range(1, len(final_dataset)+1))
    final_dataset['Sr. No.'] = range(1, len(final_dataset)+1)

    # Efficiencies and CEF
    with np.errstate(divide='ignore', invalid='ignore'):
        final_dataset['Coulombic_Efficiency'] = final_dataset['Discharge_Capacity'] / final_dataset['Charge_Capacity']
        final_dataset['Energy_Efficiency'] = final_dataset['Discharge_Energy'] / final_dataset['Charge_Energy']
        CE = final_dataset['Coulombic_Efficiency'].astype(float)
        EE = final_dataset['Energy_Efficiency'].astype(float)
        final_dataset['CEF'] = 2 / (1 / np.exp(-10 * (1 - CE)) + 1 / np.exp(-10 * (1 - EE)))

    # Optionally drop the first paired cycle
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
    df, _ = canonicalize_columns(df, ALIAS_MAP)
    df = _normalize_units(df)
    return df, fmt

def validate_processed_dataset(uploaded_file):
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
        df["CEF"] = 2 / (1 / np.exp(-10 * (1 - CE)) + 1 / np.exp(-10 * (1 - EE))]
        notes.append("CEF computed from CE and EE.")

    # Validate coverage
    has_ok = ("CEF" in df.columns) or ({"Coulombic_Efficiency","Energy_Efficiency"}.issubset(df.columns)) \
             or ({"Discharge_Capacity","Charge_Capacity","Discharge_Energy","Charge_Energy"}.issubset(df.columns))
    if not has_ok:
        raise ValueError("Processed dataset lacks required columns. Provide CEF or CE+EE or full capacity/energy pairs.")

    # Coerce numeric
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
