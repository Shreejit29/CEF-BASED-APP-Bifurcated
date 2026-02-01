# main.py  (PHASE I ONLY – CLEAN)

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
# App Config
# -------------------------------------------------
st.set_page_config(
    page_title="EFRS – Phase I (CEF Analysis)",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 EFRS – Phase I")
st.markdown(
    "**Per-cycle battery efficiency analysis (CE, EE, CEF)**  \n"
    "_No slopes, no ML, no prediction – physics only_"
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

            st.subheader("Per-Cycle Dataset (Phase I Output)")
            st.write(f"Rows (cycles): {len(final_dataset)}")
            st.dataframe(final_dataset.head(10))

            # ---------------- PLOTS ----------------
            st.subheader("📈 CEF vs Cycle Number")
            st.plotly_chart(
                cef_figure({"first_n_df": final_dataset}),
                use_container_width=True
            )

            st.subheader("📊 Efficiencies & Capacity Trends")
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

            # ---------------- DOWNLOADS ----------------
            st.subheader("💾 Download Results")

            csv_data = final_dataset.to_csv(index=False)
            st.download_button(
                "Download per-cycle dataset (CSV)",
                csv_data,
                file_name=f"phase1_per_cycle_{uploaded_file.name}.csv",
                mime="text/csv"
            )

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                final_dataset.to_excel(writer, sheet_name="Phase_I_Per_Cycle", index=False)

            st.download_button(
                "Download per-cycle dataset (Excel)",
                excel_buffer.getvalue(),
                file_name=f"phase1_per_cycle_{uploaded_file.name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")

else:
    st.info("👆 Upload raw or processed battery data to begin Phase I analysis.")
