import pandas as pd
import numpy as np
from io import BytesIO
from utils import time_to_decimal_hours

# existing: REQUIRED_COLS, load_excel_first_sheet, build_final_dataset ...

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
