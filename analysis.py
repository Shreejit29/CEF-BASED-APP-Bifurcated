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
    # Choose columns flexibly
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

    # Drop rows with NaN CEF or Cycle
    first = first[[cyc_col, cef_col]].dropna()
    if len(first) == 0:
        return result

    y = first[cef_col].to_numpy(dtype=float)
    x_vals = first[cyc_col].to_numpy(dtype=float)
    x = x_vals.reshape(-1, 1)

    # Stats with safety
    s = safe_stats(y)
    result.update(s)

    # Only fit when at least 2 distinct x and 2 samples
    if len(y) >= 2 and np.unique(x_vals).size >= 2:
        model = LinearRegression()
        model.fit(x, y)
        result["slope"] = float(model.coef_[0])
        result["trend_y"] = model.predict(x).astype(float)
    else:
        result["slope"] = None
        result["trend_y"] = None

    # Return the same columns as upstream expects for plotting (rename to canonical for convenience)
    out = first.rename(columns={cyc_col: "Cycle_Number", cef_col: "CEF"})
    result["first_n_df"] = out
    return result
