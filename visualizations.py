import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

def plot_cef(df, first_n=10):
    df_n = df.head(first_n)
    X = df_n['Cycle_Number'].values.reshape(-1,1)
    y = df_n['CEF'].values
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_n['Cycle_Number'], y=y,
                             mode='lines+markers', name='CEF'))
    fig.add_trace(go.Scatter(x=df_n['Cycle_Number'],
                             y=model.predict(X),
                             mode='lines', name=f'Trend (slope {slope:.6f})',
                             line=dict(dash='dash', color='red')))
    fig.update_layout(title="CEF vs Cycle Number",
                      xaxis_title="Cycle Number", yaxis_title="CEF")
    return fig, slope

def plot_efficiencies(df, first_n=10):
    df_n = df.head(first_n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_n['Cycle_Number'], y=df_n['Coulombic_Efficiency'],
                             mode='lines+markers', name='Coulombic Efficiency'))
    fig.add_trace(go.Scatter(x=df_n['Cycle_Number'], y=df_n['Energy_Efficiency'],
                             mode='lines+markers', name='Energy Efficiency'))
    fig.update_layout(title="Efficiency Trends",
                      xaxis_title="Cycle Number", yaxis_title="Efficiency")
    return fig

def plot_capacities(df, first_n=10):
    df_n = df.head(first_n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_n['Cycle_Number'], y=df_n['Charge_Capacity'],
                             mode='lines+markers', name='Charge Capacity'))
    fig.add_trace(go.Scatter(x=df_n['Cycle_Number'], y=df_n['Discharge_Capacity'],
                             mode='lines+markers', name='Discharge Capacity'))
    fig.update_layout(title="Capacity Trends",
                      xaxis_title="Cycle Number", yaxis_title="Capacity (mAh)")
    return fig
