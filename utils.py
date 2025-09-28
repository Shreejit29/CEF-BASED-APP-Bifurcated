from datetime import datetime
import numpy as np
import re
from typing import Dict, List, Tuple
import pandas as pd

def time_to_decimal_hours(time_str):
    """
    Convert 'HH:MM:SS' or 'HH:MM:SS.sss' to decimal hours.
    Accepts strings or pandas/Excel time-like objects.
    """
    s = str(time_str)
    # Try multiple precise formats
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            t = datetime.strptime(s, fmt)
            return t.hour + t.minute/60 + t.second/3600 + t.microsecond/3600000000
        except Exception:
            continue
    # Fallback: try pandas to_datetime
    try:
        t = pd.to_datetime(s, errors="coerce")
        if pd.isna(t):
            raise ValueError
        return t.hour + t.minute/60 + t.second/3600 + t.microsecond/3600000000
    except Exception:
        # Final fallback: return NaN for bad inputs
        return np.nan

def safe_stats(values):
    n = len(values)
    if n == 0:
        return {"range": None, "std": None, "mean": None}
    arr = np.asarray(values, dtype=float)
    rng = float(np.nanmax(arr) - np.nanmin(arr)) if n >= 1 else None
    std = float(np.nanstd(arr, ddof=1)) if n >= 2 else 0.0
    mean = float(np.nanmean(arr)) if n >= 1 else None
    return {"range": rng, "std": std, "mean": mean}

# ---------------- Column canonicalization ----------------

def _norm(s: str) -> str:
    """
    Normalize a column name for matching:
    - strip whitespace
    - lowercase
    - remove spaces/underscores/hyphens
    - drop non-alphanumeric except (), %
    """
    s = str(s).strip().lower()
    s = re.sub(r"[ _\-]+", "", s)
    s = re.sub(r"[^a-z0-9()%]", "", s)
    return s

def canonicalize_columns(df: pd.DataFrame, alias_map: Dict[str, List[str]]) -> Tuple[pd.DataFrame, list]:
    """
    Rename df columns to canonical keys using alias_map.
    alias_map keys: canonical names used by the pipeline.
    alias_map values: list of acceptable variants/aliases.

    Returns:
      (renamed_df, notes)
    """
    reverse = {}
    for canon, aliases in alias_map.items():
        for a in [canon] + aliases:
            reverse[_norm(a)] = canon

    rename = {}
    for col in df.columns:
        key = _norm(col)
        if key in reverse:
            rename[col] = reverse[key]

    notes = []
    if rename:
        notes.append("Canonicalized: " + ", ".join([f\"{k}->{v}\" for k, v in rename.items()]))

    return df.rename(columns=rename), notes
