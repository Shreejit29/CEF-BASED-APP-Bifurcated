from datetime import datetime
import numpy as np
import re
from typing import Dict, List, Tuple
import pandas as pd

def time_to_decimal_hours(time_str):
    t = datetime.strptime(str(time_str), '%H:%M:%S.%f')
    return t.hour + t.minute/60 + t.second/3600 + t.microsecond/3600000000

def safe_stats(values):
    n = len(values)
    if n == 0:
        return {"range": None, "std": None, "mean": None}
    arr = np.asarray(values, dtype=float)
    rng = float(arr.max() - arr.min()) if n >= 1 else None
    std = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    mean = float(np.mean(arr)) if n >= 1 else None
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
        notes.append("Canonicalized: " + ", ".join([f"{k}->{v}" for k, v in rename.items()]))

    return df.rename(columns=rename), notes
