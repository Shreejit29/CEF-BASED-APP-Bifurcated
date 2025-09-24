import plotly.graph_objects as go

def cef_figure(stats):
    df = stats["first_n_df"]
    fig = go.Figure()
    if len(df) == 0:
        fig.update_layout(title="CEF vs Cycle Number (no data)",
                          xaxis_title="Cycle Number", yaxis_title="CEF Value")
        return fig
    fig.add_trace(go.Scatter(
        x=df["Cycle_Number"], y=df["CEF"],
        mode="lines+markers", name="CEF",
        line=dict(color="blue", width=2), marker=dict(size=8)
    ))
    if stats["trend_y"] is not None:
        fig.add_trace(go.Scatter(
            x=df["Cycle_Number"], y=stats["trend_y"],
            mode="lines", name=f"Trend",
            line=dict(color="red", width=2, dash="dash")
        ))
    fig.update_layout(title="CEF vs Cycle Number (First 10 Cycles)",
                      xaxis_title="Cycle Number", yaxis_title="CEF Value",
                      hovermode="x")
    return fig

def efficiencies_figure(stats):
    df = stats["first_n_df"]
    fig = go.Figure()
    if len(df) == 0:
        fig.update_layout(title="Efficiency Trends",
                          xaxis_title="Cycle Number", yaxis_title="Efficiency")
        return fig
    fig.add_trace(go.Scatter(
        x=df["Cycle_Number"], y=df["Coulombic_Efficiency"],
        mode="lines+markers", name="Coulombic Efficiency",
        line=dict(color="green")
    ))
    fig.add_trace(go.Scatter(
        x=df["Cycle_Number"], y=df["Energy_Efficiency"],
        mode="lines+markers", name="Energy Efficiency",
        line=dict(color="orange")
    ))
    fig.update_layout(title="Efficiency Trends",
                      xaxis_title="Cycle Number", yaxis_title="Efficiency")
    return fig

def capacities_figure(stats):
    df = stats["first_n_df"]
    fig = go.Figure()
    if len(df) == 0:
        fig.update_layout(title="Capacity Trends",
                          xaxis_title="Cycle Number", yaxis_title="Capacity (mAh)")
        return fig
    fig.add_trace(go.Scatter(
        x=df["Cycle_Number"], y=df["Charge_Capacity"],
        mode="lines+markers", name="Charge Capacity",
        line=dict(color="purple")
    ))
    fig.add_trace(go.Scatter(
        x=df["Cycle_Number"], y=df["Discharge_Capacity"],
        mode="lines+markers", name="Discharge Capacity",
        line=dict(color="brown")
    ))
    fig.update_layout(title="Capacity Trends",
                      xaxis_title="Cycle Number", yaxis_title="Capacity (mAh)")
    return fig
