import streamlit as st
import io
import pandas as pd
import numpy as np
from processing import load_excel_first_sheet, build_final_dataset, validate_processed_dataset, _read_any
from analysis import first_n_cef_stats
from viz import cef_figure, efficiencies_figure, capacities_figure

st.set_page_config(page_title="Battery Health Prediction - CEF Analysis", page_icon="🔋", layout="wide")
st.title("🔋 Battery Health Prediction - CEF Analysis")
st.markdown("Upload raw cycler data for full processing or upload an already processed dataset to compute CEF statistics and plots")

st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Choose input type",
    ("Raw cycler Excel", "Processed dataset"),
    help="Raw cycler Excel runs full cleaning and feature engineering. Processed dataset skips to analysis if required columns exist."
)

remove_first_row = st.sidebar.checkbox(
    "Remove First Row (Conditioning Cycle)", value=True,
    help="Applies only to raw cycler Excel processing"
)

def compute_derivables(df: pd.DataFrame):
    if "Coulombic_Efficiency" not in df.columns and {"Discharge_Capacity","Charge_Capacity"}.issubset(df.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            df["Coulombic_Efficiency"] = np.where(df["Charge_Capacity"]>0, df["Discharge_Capacity"]/df["Charge_Capacity"], np.nan)
    if "Energy_Efficiency" not in df.columns and {"Discharge_Energy","Charge_Energy"}.issubset(df.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            df["Energy_Efficiency"] = np.where(df["Charge_Energy"]>0, df["Discharge_Energy"]/df["Charge_Energy"], np.nan)
    if "CEF" not in df.columns and {"Coulombic_Efficiency","Energy_Efficiency"}.issubset(df.columns):
        CE = df["Coulombic_Efficiency"].astype(float)
        EE = df["Energy_Efficiency"].astype(float)
        df["CEF"] = 2 / (1 / np.exp(-10*(1-CE)) + 1 / np.exp(-10*(1-EE)))
    if "Cycle_Number" not in df.columns:
        df = df.reset_index(drop=True)
        df.insert(0, "Cycle_Number", range(1, len(df)+1))
    return df

uploaded_file = st.file_uploader(
    "Upload file",
    type=(['xlsx','xls'] if mode == "Raw cycler Excel" else ['csv','xlsx','xls'])
)

if uploaded_file is not None:
    try:
        final_dataset = None

        if mode == "Raw cycler Excel":
            sheet_name, df_raw = load_excel_first_sheet(uploaded_file)
            st.success(f"File uploaded successfully! Sheet: {sheet_name}")

            st.subheader("📊 Original Data Preview")
            st.write(f"Dataset shape: {df_raw.shape}")
            st.dataframe(df_raw.head())

            with st.expander("🧪 Data Preparation (edit, map, and clean)", expanded=True):
                st.caption("Edit cells, add or delete rows before processing.")
                editable = st.data_editor(df_raw, num_rows="dynamic", use_container_width=True)
                can_process = all(k in editable.columns for k in ["Time","Date","Current (mA)","Capacity (mAh)","Energy (mWh)"])
                if not can_process:
                    st.error("For raw processing, ensure columns: Time, Date, Current (mA), Capacity (mAh), Energy (mWh).")
                else:
                    with st.spinner("Processing raw data..."):
                        final_dataset = build_final_dataset(editable, remove_first_row)

        else:
            df_loaded, fmt = _read_any(uploaded_file)
            st.success(f"Processed dataset loaded ({fmt}).")

            with st.expander("🧪 Data Preparation (edit and compute missing fields)", expanded=True):
                editable = st.data_editor(df_loaded, num_rows="dynamic", use_container_width=True)
                df_mapped = editable.copy()  # optional mapping could be added here
                df_ready = compute_derivables(df_mapped)
                try:
                    # Try strict validator first
                    final_dataset, notes = validate_processed_dataset(io.BytesIO(editable.to_csv(index=False).encode())) \
                        if fmt == "csv" else validate_processed_dataset(uploaded_file)
                except Exception:
                    # Fall back to the editor-derived frame (best-effort)
                    final_dataset = df_ready
                    notes = ["Used edited data with computed fields (best-effort)."]
                if notes:
                    st.warning("Notes: " + " | ".join(notes))

        if final_dataset is None or final_dataset.empty:
            st.warning("No valid rows available for analysis after preparation. Please adjust the data and try again.")
        else:
            st.subheader("🔧 Data Preview For Analysis")
            st.write(f"Final dataset shape: {final_dataset.shape}")
            st.dataframe(final_dataset.head(10))

            st.subheader("📈 CEF Analysis - First 10 Cycles")
            stats = first_n_cef_stats(final_dataset, first_n=10)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("CEF Slope", f"{stats['slope']:.6f}" if stats['slope'] is not None else "N/A", help="Linear regression slope")
            col2.metric("CEF Range", f"{stats['range']:.6f}" if stats['range'] is not None else "N/A", help="Max - Min CEF values")
            col3.metric("CEF Std Dev", f"{stats['std']:.6f}" if stats['std'] is not None else "N/A", help="Standard deviation")
            col4.metric("CEF Mean", f"{stats['mean']:.6f}" if stats['mean'] is not None else "N/A", help="Average CEF value")

            st.plotly_chart(cef_figure(stats), use_container_width=True)

            st.subheader("📊 Additional Analysis")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(efficiencies_figure(stats), use_container_width=True)
            with c2:
                st.plotly_chart(capacities_figure(stats), use_container_width=True)

            st.subheader("💾 Download Results")
            statistics_df = pd.DataFrame({
                'Parameter': ['CEF Slope (Linear Regression)', 'CEF Range', 'CEF Standard Deviation', 'CEF Mean'],
                'Value': [stats['slope'], stats['range'], stats['std'], stats['mean']],
                'Description': [
                    'Linear regression slope of CEF vs Cycle Number for first 10 cycles',
                    'Difference between maximum and minimum CEF values in first 10 cycles',
                    'Sample standard deviation of CEF values for first 10 cycles',
                    'Mean CEF value for first 10 cycles'
                ]
            })
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                statistics_df.to_excel(writer, sheet_name='CEF_Statistics', index=False)
                stats['first_n_df'].to_excel(writer, sheet_name='First_10_Cycles', index=False)
                final_dataset.to_excel(writer, sheet_name='Complete_Dataset', index=False)
            st.download_button("📥 Download Complete Analysis (Excel)", output.getvalue(),
                               file_name="CEF_Analysis_Results.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("📄 Download Dataset (CSV)", final_dataset.to_csv(index=False),
                               file_name="processed_battery_data.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("👆 Upload a file to begin analysis")
