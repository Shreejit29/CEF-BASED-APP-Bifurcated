# main.py  (PHASE I + PHASE II + PHASE III + PHASE IV – CLEAN)

import streamlit as st
import io
import pandas as pd
import numpy as np

from processing import (
    load_excel_first_sheet,
    build_final_dataset,
    _read_any,
    validate_processed_dataframe,
    ALIAS_MAP
)
from viz import cef_figure, efficiencies_figure, capacities_figure
from utils import canonicalize_columns
from predictor_helpers import (
    compute_D1_cef_decay_rate,
    compute_D2_cef_instability,
    compute_D3_cef_acceleration,
    compute_predictor_index
)

# -------------------------------------------------
# Phase II – Early Window Helper
# -------------------------------------------------
def extract_early_window(df_cycles: pd.DataFrame, n_cycles: int) -> pd.DataFrame:
    if df_cycles is None or df_cycles.empty:
        return df_cycles
    df_sorted = df_cycles.sort_values("Cycle_Number")
    return df_sorted.head(n_cycles).reset_index(drop=True)

# -------------------------------------------------
# Phase III – EFRS Score Helper
# -------------------------------------------------
def compute_efrs_score(early_window_df: pd.DataFrame) -> float:
    if early_window_df is None or early_window_df.empty:
        return np.nan
    if "CEF" not in early_window_df.columns:
        return np.nan
    cef = pd.to_numeric(early_window_df["CEF"], errors="coerce").dropna()
    if cef.empty:
        return np.nan
    efrs = 100.0 * float(cef.mean())
    return round(max(0.0, min(100.0, efrs)), 2)

# -------------------------------------------------
# App Config
# -------------------------------------------------
st.set_page_config(
    page_title="EFRS – Phase I to IV",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 EFRS – Phase I → IV")
st.markdown(
    "**Phase I:** CE, EE, CEF (physics)  \n"
    "**Phase II:** Early window  \n"
    "**Phase III:** EFRS (0–100)  \n"
    "**Phase IV:** Predictive descriptors (D1–D3)"
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Choose input type",
    ("Raw cycler Excel", "Processed dataset"),
)

remove_first_row = st.sidebar.checkbox(
    "Remove First Cycle (formation / conditioning)",
    value=True
)

st.sidebar.header("Phase II – Early Window")
early_n = st.sidebar.slider(
    "Number of early cycles",
    min_value=1,
    max_value=50,
    value=10,
    step=1
)

# -------------------------------------------------
# File Upload
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload files",
    type=(['xlsx', 'xls'] if mode == "Raw cycler Excel" else ['csv', 'xlsx', 'xls']),
    accept_multiple_files=True
)

# -------------------------------------------------
# Main Loop
# -------------------------------------------------
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            st.write(f"## 📁 File: {uploaded_file.name}")

            # ---------------- RAW DATA ----------------
            if mode == "Raw cycler Excel":
                sheet_name, df_raw = load_excel_first_sheet(uploaded_file)
                st.success(f"Loaded sheet: {sheet_name}")
                st.dataframe(df_raw.head())
                with st.spinner("Processing raw cycler data → per-cycle metrics"):
                    final_dataset = build_final_dataset(df_raw, remove_first_row)

            # ---------------- PROCESSED DATA ----------------
            else:
                df_loaded, fmt = _read_any(uploaded_file)
                st.success(f"Loaded processed dataset ({fmt})")
                df_loaded, canon_notes = canonicalize_columns(df_loaded, ALIAS_MAP)
                if canon_notes:
                    st.info(" | ".join(canon_notes))
                final_dataset, notes = validate_processed_dataframe(df_loaded)
                if notes:
                    st.warning("Notes: " + " | ".join(notes))

            # ---------------- VALIDATION ----------------
            if final_dataset is None or final_dataset.empty:
                st.warning("No valid per-cycle data available.")
                continue

            # ---------------- PHASE I ----------------
            st.subheader("📊 Phase I – Per-Cycle Dataset")
            st.dataframe(final_dataset.head(10), use_container_width=True)

            st.plotly_chart(
                cef_figure({"first_n_df": final_dataset}),
                use_container_width=True
            )

            # ---------------- PHASE II ----------------
            early_window_df = extract_early_window(final_dataset, early_n)

            st.subheader("🔍 Phase II – Early Window")
            st.dataframe(early_window_df, use_container_width=True)

            # ---------------- PHASE III ----------------
            efrs_score = compute_efrs_score(early_window_df)

            st.subheader("🎯 Phase III – EFRS")
            st.metric("EFRS (0–100)", f"{efrs_score:.2f}" if not np.isnan(efrs_score) else "NA")

            # ---------------- PHASE IV ----------------
            D1 = compute_D1_cef_decay_rate(early_window_df)
            D2 = compute_D2_cef_instability(early_window_df)
            D3 = compute_D3_cef_acceleration(early_window_df)
            predictor_index = compute_predictor_index(efrs_score, D1, D2, D3)

            st.subheader("🧠 Phase IV – Predictive Descriptors")

            c1, c2, c3 = st.columns(3)
            c1.metric("D1: dCEF/dN", f"{D1:.3e}" if not np.isnan(D1) else "NA")
            c2.metric("D2: σ(CEF)", f"{D2:.3e}" if not np.isnan(D2) else "NA")
            c3.metric("D3: d²CEF/dN²", f"{D3:.3e}" if not np.isnan(D3) else "NA")

            st.subheader("🚨 Predictor Output")
            st.metric(
                "Early Failure Predictor Index",
                f"{predictor_index:.2f}" if not np.isnan(predictor_index) else "NA"
            )

            # ---------------- DOWNLOADS ----------------
            st.subheader("💾 Download Results")

            early_out = early_window_df.copy()
            early_out["EFRS"] = efrs_score
            early_out["D1"] = D1
            early_out["D2"] = D2
            early_out["D3"] = D3
            early_out["Predictor_Index"] = predictor_index

            st.download_button(
                "Download early-window + predictors (CSV)",
                early_out.to_csv(index=False),
                file_name=f"efrs_predictor_{uploaded_file.name}.csv",
                mime="text/csv"
            )

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                final_dataset.to_excel(writer, sheet_name="Phase_I", index=False)
                early_window_df.to_excel(writer, sheet_name="Phase_II", index=False)
                pd.DataFrame({"EFRS": [efrs_score]}).to_excel(writer, sheet_name="Phase_III", index=False)
                early_out.to_excel(writer, sheet_name="Phase_IV", index=False)

            st.download_button(
                "Download Phase I–IV (Excel)",
                excel_buffer.getvalue(),
                file_name=f"efrs_phase1_to_phase4_{uploaded_file.name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")

else:
    st.info("👆 Upload raw or processed battery data to begin EFRS analysis.")
