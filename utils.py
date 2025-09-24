from datetime import datetime
import numpy as np

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
