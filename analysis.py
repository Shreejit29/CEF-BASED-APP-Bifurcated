import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import safe_stats

def first_n_cef_stats(final_dataset: pd.DataFrame, first_n=10):
    first = final_dataset.head(first_n).copy()
    result = {
        "first_n_df": first,
        "slope": None,
        "range": None,
        "std": None,
        "mean": None,
        "trend_y": None
    }
    if len(first) == 0 or "CEF" not in first.columns or "Cycle_Number" not in first.columns:
        return result

    y = first["CEF"].to_numpy()
    x = first["Cycle_Number"].to_numpy().reshape(-1, 1)
    s = safe_stats(y)
    result.update(s)

    # Fit only when we have at least 2 distinct x values
    if len(first) >= 2 and len(np.unique(x)) >= 2:
        model = LinearRegression()
        model.fit(x, y)
        result["slope"] = float(model.coef_[0])
        result["trend_y"] = model.predict(x).astype(float)
    else:
        result["slope"] = None
        result["trend_y"] = None

    return result
