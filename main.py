import streamlit as st
import io
import pandas as pd
from processing import load_excel_first_sheet, build_final_dataset, validate_processed_dataset
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

if mode == "Raw cycler Excel":
    uploaded_file = st.file_uploader("Upload raw cycler Excel", type=['xlsx', 'xls'])
else:
    uploaded_file = st.file_uploader("Upload processed dataset (CSV or Excel)", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        if mode == "Raw cycler Excel":
            sheet_name, df_raw = load_excel_first_sheet(uploaded_file)
            st.success(f"File uploaded successfully! Sheet: {sheet_name}")
            st.subheader("📊 Original Data Preview")
            st.write(f"Dataset shape: {df_raw.shape}")
            st.dataframe(df_raw.head())
            with st.spinner("Processing raw data..."):
                final_dataset = build_final_dataset(df_raw, remove_first_row)
        else:
            st.info("Expecting columns like Cycle_Number, Charge/Discharge Capacity/Energy, CE/EE, and/or CEF. Missing derivable columns will be recomputed if possible.")
            final_dataset, notes = validate_processed_dataset(uploaded_file)
            if notes:
                st.warning("Notes: " + " | ".join(notes))
            st.success("Processed dataset loaded.")

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
        processed_data = output.getvalue()

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                label="📥 Download Complete Analysis (Excel)",
                data=processed_data,
                file_name="CEF_Analysis_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with d2:
            st.download_button(
                label="📄 Download Dataset (CSV)",
                data=final_dataset.to_csv(index=False),
                file_name="processed_battery_data.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("👆 Upload a file to begin analysis")
def schema_mapper(df: pd.DataFrame, required_map, optional_map):
    st.subheader("🧭 Column Mapping")
    st.caption("Map columns from the uploaded data to expected fields. Unmapped optional fields are okay.")
    cols = ["-- none --"] + list(df.columns)
    mapped = {}
    with st.container():
        st.markdown("Required fields")
        rcols = st.columns(2)
        idx = 0
        for key, label in required_map.items():
            with rcols[idx % 2]:
                choice = st.selectbox(f"{label}", cols, key=f"req_{key}")
                mapped[key] = None if choice == "-- none --" else choice
            idx += 1
    with st.container():
        st.markdown("Optional fields")
        ocols = st.columns(2)
        idx = 0
        for key, label in optional_map.items():
            with ocols[idx % 2]:
                choice = st.selectbox(f"{label}", cols, key=f"opt_{key}")
                mapped[key] = None if choice == "-- none --" else choice
            idx += 1
    return mapped

def apply_mapping(df: pd.DataFrame, mapped: dict):
    # Create a copy with unified column names based on mapping
    out = df.copy()
    rename_dict = {v: k for k, v in mapped.items() if v is not None}
    # Only rename existing columns
    rename_dict = {old: new for old, new in rename_dict.items() if old in out.columns}
    out = out.rename(columns=rename_dict)
    return out

def compute_derivables(df: pd.DataFrame):
    # Compute CE/EE/CEF if derivable
    if "Coulombic_Efficiency" not in df and {"Discharge_Capacity","Charge_Capacity"}.issubset(df.columns):
        df["Coulombic_Efficiency"] = df["Discharge_Capacity"] / df["Charge_Capacity"]
    if "Energy_Efficiency" not in df and {"Discharge_Energy","Charge_Energy"}.issubset(df.columns):
        df["Energy_Efficiency"] = df["Discharge_Energy"] / df["Charge_Energy"]
    if "CEF" not in df and {"Coulombic_Efficiency","Energy_Efficiency"}.issubset(df.columns):
        CE = df["Coulombic_Efficiency"].astype(float)
        EE = df["Energy_Efficiency"].astype(float)
        df["CEF"] = 2 / (1 / ( (np.exp(-10 * (1 - CE))) ) + 1 / ( (np.exp(-10 * (1 - EE))) ))
    if "Cycle_Number" not in df:
        df = df.reset_index(drop=True)
        df.insert(0, "Cycle_Number", range(1, len(df)+1))
    return df

# After file upload and initial loading (df_raw or final_dataset from processed), insert Data Preparation:
if uploaded_file is not None:
    try:
        if mode == "Raw cycler Excel":
            sheet_name, df_raw = load_excel_first_sheet(uploaded_file)
            st.success(f"File uploaded successfully! Sheet: {sheet_name}")
            st.subheader("📊 Original Data Preview")
            st.dataframe(df_raw.head())

            with st.expander("🧪 Data Preparation (edit, map, and clean)", expanded=True):
                st.caption("Edit cells, add or delete rows, and map columns to expected names before processing.")
                editable = st.data_editor(df_raw, num_rows="dynamic", use_container_width=True)
                required_map = {
                    "Time": "Time column (HH:MM:SS.sss)",
                    "Date": "Date column",
                    "Current (mA)": "Current (mA)",
                    "Capacity (mAh)": "Capacity (mAh)",
                    "Energy (mWh)": "Energy (mWh)"
                }
                optional_map = {
                    "Sr. No.": "Sr. No.",
                    "Voltage (mV)": "Voltage (mV)"
                }
                mapping = schema_mapper(editable, required_map, optional_map)
                df_mapped = apply_mapping(editable, mapping)

                # Minimal validation for raw path: need Time, Date, Current/Capacity/Energy
                missing_req = [k for k in required_map if k not in df_mapped.columns]
                if missing_req:
                    st.error(f"Missing required fields after mapping: {missing_req}")
                else:
                    proceed = st.checkbox("Use edited & mapped data for processing", value=True)
                    if proceed:
                        with st.spinner("Processing raw data..."):
                            final_dataset = build_final_dataset(df_mapped, remove_first_row)
        else:
            st.info("Upload processed dataset, then edit/map as needed before analysis.")
            df_loaded, _ = processing._read_any(uploaded_file)  # or replicate simple reader here
            with st.expander("🧪 Data Preparation (edit, map, and clean)", expanded=True):
                editable = st.data_editor(df_loaded, num_rows="dynamic", use_container_width=True)
                # For processed, mapping keys are the unified names the pipeline understands
                required_any = ["CEF"]
                optional_map = {
                    "Cycle_Number": "Cycle_Number",
                    "Charge_Capacity": "Charge_Capacity",
                    "Discharge_Capacity": "Discharge_Capacity",
                    "Charge_Energy": "Charge_Energy",
                    "Discharge_Energy": "Discharge_Energy",
                    "Coulombic_Efficiency": "Coulombic_Efficiency",
                    "Energy_Efficiency": "Energy_Efficiency"
                }
                # Provide one-to-one mapping widget only for optional since processed schema varies widely
                mapping = {k: st.selectbox(f"Map {k} from", ["-- none --"] + list(editable.columns), index=0, key=f"pmap_{k}")
                           for k in optional_map.keys()}
                mapped = {}
                for k, v in mapping.items():
                    mapped[k] = None if v == "-- none --" else v
                df_mapped = apply_mapping(editable, mapped)
                df_mapped = compute_derivables(df_mapped)

                if not any(col in df_mapped.columns for col in ["CEF", "Coulombic_Efficiency", "Charge_Capacity"]):
                    st.error("Please map at least CEF or CE/EE or capacities/energies.")
                else:
                    final_dataset = df_mapped.copy()

        # Continue with the existing analysis and downloads using final_dataset
        if 'final_dataset' in locals():
            st.subheader("🔧 Data Preview For Analysis")
            st.dataframe(final_dataset.head(10))
            # ... existing stats, charts, and downloads ...
