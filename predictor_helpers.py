# predictor_helpers.py

import numpy as np
import pandas as pd

# -------------------------------------------------
# D1: CEF decay rate (dCEF / dN)
# -------------------------------------------------
def compute_D1_cef_decay_rate(df_early: pd.DataFrame) -> float:
    """
    Linear decay rate of CEF across early cycles.
    """
    if df_early is None or len(df_early) < 2:
        return np.nan

    x = df_early["Cycle_Number"].values.astype(float)
    y = df_early["CEF"].values.astype(float)

    coeffs = np.polyfit(x, y, deg=1)
    return float(coeffs[0])  # slope


# -------------------------------------------------
# D2: CEF instability (local variance)
# -------------------------------------------------
def compute_D2_cef_instability(df_early: pd.DataFrame) -> float:
    """
    Standard deviation of CEF in early window.
    """
    if df_early is None or df_early.empty:
        return np.nan

    return float(df_early["CEF"].std())


# -------------------------------------------------
# D3: CEF acceleration (smoothed curvature)
# -------------------------------------------------
def compute_D3_cef_acceleration(df_early: pd.DataFrame) -> float:
    """
    Mean second difference of CEF (numerically stable).
    """
    if df_early is None or len(df_early) < 3:
        return np.nan

    cef = df_early["CEF"].values.astype(float)
    second_diff = np.diff(cef, n=2)
    return float(np.mean(second_diff))


# -------------------------------------------------
# Simple deterministic predictor (no ML)
# -------------------------------------------------
def compute_predictor_index(efrs: float, D1: float, D2: float, D3: float) -> float:
    """
    Physics-inspired early failure risk index.
    Higher = worse.
    """
    if np.isnan(efrs):
        return np.nan

    # Normalized, sign-aware combination
    risk_index = (
        (100.0 - efrs) +      # low EFRS → higher risk
        (-500.0 * D1) +       # fast decay → higher risk
        (200.0 * D2) +        # instability → higher risk
        (-200.0 * D3)         # accelerating drop → higher risk
    )

    return round(float(risk_index), 2)
