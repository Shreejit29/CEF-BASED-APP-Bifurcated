import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import safe_stats

# Accept common aliases for robustness
CEF_ALIASES = ["CEF", "cef", "cef_value", "CEF value"]
CYCLE_ALIASES = ["Cycle_Number", "Cycle", "Cycle No", "cycle number", "Cycle_Index", "Cycle_index", "Cycle Index"]

def _first_existing(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def first_n_cef_stats(final_dataset: pd.DataFrame, first_n=10):
    # Flexible column resolution
    cef_col = _first_existing(final_dataset, CEF_ALIASES)
    cyc_col = _first_existing(final_dataset, CYCLE_ALIASES)

    first = final_dataset.head(first_n).copy()
    result = {
        "first_n_df": first if cef_col and cyc_col else first,
        "slope": None,
        "range": None,
        "std": None,
        "mean": None,
        "trend_y": None
    }

    if len(first) == 0 or cef_col is None or cyc_col is None:
        return result

    # Collect optional plotting columns if present
    opt_cols = []
    for c in ["Coulombic_Efficiency", "Energy_Efficiency", "Charge_Capacity", "Discharge_Capacity"]:
        if c in first.columns:
            opt_cols.append(c)

    # Drop rows missing essential fields for CEF trend
    core = first[[cyc_col, cef_col]].dropna()
    if len(core) == 0:
        out = first.rename(columns={cyc_col: "Cycle_Number", cef_col: "CEF"}) if cef_col in first.columns and cyc_col in first.columns else first
        keep_cols = ["Cycle_Number", "CEF"] + [c for c in opt_cols if c in out.columns]
        result["first_n_df"] = out[keep_cols] if set(keep_cols).issubset(out.columns) else out
        return result

    y = core[cef_col].to_numpy(dtype=float)
    x_vals = core[cyc_col].to_numpy(dtype=float)
    x = x_vals.reshape(-1, 1)

    # Compute descriptive stats with consistent definitions
    # safe_stats may already compute mean and std, but enforce ddof=1 and range explicitly
    mean_val = float(np.mean(y)) if y.size else None
    std_val = float(np.std(y, ddof=1)) if y.size >= 2 else None
    range_val = float(np.nanmax(y) - np.nanmin(y)) if y.size else None

    result["mean"] = mean_val
    result["std"] = std_val
    result["range"] = range_val

    # Linear trend if possible
    if len(y) >= 2 and np.unique(x_vals).size >= 2:
        model = LinearRegression()
        model.fit(x, y)
        result["slope"] = float(model.coef_[0])
        result["trend_y"] = model.predict(x).astype(float)
    else:
        result["slope"] = None
        result["trend_y"] = None

    # Build output for plotting with optional columns
    merged = core.copy()
    if opt_cols:
        merged = merged.merge(first[[cyc_col] + opt_cols], on=cyc_col, how="left", suffixes=("", ""))

    out = merged.rename(columns={cyc_col: "Cycle_Number", cef_col: "CEF"})
    keep_cols = ["Cycle_Number", "CEF"] + [c for c in ["Coulombic_Efficiency", "Energy_Efficiency", "Charge_Capacity", "Discharge_Capacity"] if c in out.columns]
    result["first_n_df"] = out[keep_cols]

    return result
