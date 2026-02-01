# main.py  (PHASE I + PHASE II – CLEAN)

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

# -------------------------------------------------
# Phase II – Early Window Helper
# -------------------------------------------------
def extract_early_window(df_cycles: pd.DataFrame, n_cycles: int) -> pd.DataFrame:
    """
    Phase II – Early Window Extraction
    Selects the first n valid cycles.
    """
    if df_cycles is None or df_cycles.empty:
        return df_cycles

    df_sorted = df_cycles.sort_values("Cycle_Number")
    return df_sorted.head(n_cycles).reset_index(drop=True)

# -------------------------------------------------
# App Config
# -------------------------------------------------
st.set_page_config(
    page_title="EFRS – Phase I & II (CEF Analysis)",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 EFRS – Phase I & II")
st.markdown(
    "**Phase I:** Per-cycle CE, EE, CEF (physics only)  \n"
    "**Phase II:** Early-window cycle selection (no slopes, no ML)"
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

                st.subheader("Raw Data Preview")
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

            # ---------------- PHASE I OUTPUT ----------------
            st.subheader("📊 Phase I – Per-Cycle Dataset")
            st.write(f"Total cycles: {len(final_dataset)}")
            st.dataframe(final_dataset.head(10), use_container_width=True)

            st.subheader("📈 Phase I – CEF vs Cycle Number")
            st.plotly_chart(
                cef_figure({"first_n_df": final_dataset}),
                use_container_width=True
            )

            st.subheader("📊 Phase I – Efficiencies & Capacity Trends")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    efficiencies_figure({"first_n_df": final_dataset}),
                    use_container_width=True
                )
            with c2:
                st.plotly_chart(
                    capacities_figure({"first_n_df": final_dataset}),
                    use_container_width=True
                )

            # ---------------- PHASE II OUTPUT ----------------
            early_window_df = extract_early_window(final_dataset, early_n)

            st.subheader("🔍 Phase II – Early Window Dataset")
            st.caption(f"First {early_n} cycles selected (deterministic)")
            st.dataframe(early_window_df, use_container_width=True)

            st.subheader("📈 Phase II – CEF (Early Window)")
            st.plotly_chart(
                cef_figure({"first_n_df": early_window_df}),
                use_container_width=True
            )

            st.subheader("📊 Phase II – Early Window Trends")
            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(
                    efficiencies_figure({"first_n_df": early_window_df}),
                    use_container_width=True
                )
            with c4:
                st.plotly_chart(
                    capacities_figure({"first_n_df": early_window_df}),
                    use_container_width=True
                )

            # ---------------- DOWNLOADS ----------------
            st.subheader("💾 Download Results")

            csv_data = final_dataset.to_csv(index=False)
            st.download_button(
                "Download full per-cycle dataset (CSV)",
                csv_data,
                file_name=f"phase1_full_{uploaded_file.name}.csv",
                mime="text/csv"
            )

            early_csv = early_window_df.to_csv(index=False)
            st.download_button(
                "Download early-window dataset (CSV)",
                early_csv,
                file_name=f"phase2_early_window_{uploaded_file.name}.csv",
                mime="text/csv"
            )

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                final_dataset.to_excel(writer, sheet_name="Phase_I_Full", index=False)
                early_window_df.to_excel(writer, sheet_name="Phase_II_Early_Window", index=False)

            st.download_button(
                "Download Phase I & II (Excel)",
                excel_buffer.getvalue(),
                file_name=f"efrs_phase1_phase2_{uploaded_file.name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")

else:
    st.info("👆 Upload raw or processed battery data to begin Phase I & II analysis.")
