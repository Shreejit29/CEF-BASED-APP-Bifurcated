import streamlit as st
from data_processing import load_and_clean
from visualizations import plot_cef, plot_efficiencies, plot_capacities

st.set_page_config(
    page_title="Battery Health Prediction – CEF Analysis",
    page_icon="🔋", layout="wide"
)
st.title("🔋 Battery Health Prediction – CEF Analysis")

remove_first = st.sidebar.checkbox(
    "Remove First Row (Conditioning Cycle)", True
)
file = st.file_uploader("Upload Excel File", type=['xlsx','xls'])
if not file:
    st.info("👆 Upload a file to begin.")
else:
    with st.spinner("Processing…"):
        df = load_and_clean(file, remove_first)
    st.success("✅ Processing complete!")
    st.dataframe(df.head(10))

    # CEF Plot
    fig_cef, slope = plot_cef(df)
    col1, col2 = st.columns(2)
    col1.metric("CEF Slope", f"{slope:.6f}")
    st.plotly_chart(fig_cef, use_container_width=True)

    # Additional plots
    st.plotly_chart(plot_efficiencies(df), use_container_width=True)
    st.plotly_chart(plot_capacities(df), use_container_width=True)

    # (Include download buttons as before)
