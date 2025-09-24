import streamlit as st
import io
import pandas as pd
from processing import load_excel_first_sheet, build_final_dataset
from analysis import first_n_cef_stats
from viz import cef_figure, efficiencies_figure, capacities_figure

st.set_page_config(page_title="Battery Health Prediction - CEF Analysis", page_icon="🔋", layout="wide")
st.title("🔋 Battery Health Prediction - CEF Analysis")
st.markdown("Upload your battery cycler data to calculate CEF (Capacity Estimation Filter) and extract health features")

st.sidebar.header("Processing Options")
remove_first_row = st.sidebar.checkbox("Remove First Row (Conditioning Cycle)", value=True,
    help="Remove the first cycle which is typically a conditioning cycle with outlier values")

uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])

if uploaded_file is not None:
    try:
        sheet_name, df_raw = load_excel_first_sheet(uploaded_file)
        st.success(f"File uploaded successfully! Sheet: {sheet_name}")

        st.subheader("📊 Original Data Preview")
        st.write(f"Dataset shape: {df_raw.shape}")
        st.dataframe(df_raw.head())

        with st.spinner("Processing data..."):
            final_dataset = build_final_dataset(df_raw, remove_first_row)

        st.success("✅ Data processing completed!")

        st.subheader("🔧 Processed Data")
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
        st.error(f"Error processing file: {str(e)}")
        st.write("Please ensure your Excel file has the correct format with columns: Time, Date, Voltage (mV), Current (mA), Capacity (mAh), Energy (mWh)")
else:
    st.info("👆 Please upload an Excel file to begin analysis")
    st.subheader("📋 Required Data Format")
    sample_data = pd.DataFrame({
        'Sr. No.': [1, 2, 3, 4, 5],
        'Time': ['00:00:00.000', '00:00:30.000', '00:01:00.000', '00:01:30.000', '00:02:00.000'],
        'Date': ['2024-04-16 11:48:51.047'] * 5,
        'Voltage (mV)': [3708.14, 3723.11, 3730.12, 3735.12, 3740.11],
        'Current (mA)': [1274.12, 1274.11, 1274.12, 1274.11, 1274.11],
        'Capacity (mAh)': [0.18, 10.79, 21.41, 32.03, 42.64],
        'Energy (mWh)': [0.66, 40.12, 79.68, 119.30, 158.98]
    })
    st.dataframe(sample_data)
