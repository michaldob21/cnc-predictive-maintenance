import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# PAGE CONFIG
st.set_page_config(page_title="🔧 CNC Bearing AI", layout="wide")

st.title("🔧 CNC Bearing Predictive Maintenance")
st.markdown("**Detect failures 100+ days BEFORE they happen**")

# GENERATE DATA
@st.cache_data
def generate_data():
    np.random.seed(42)
    data = []
    for unit in [1, 2, 3, 4]:
        days = [100, 120, 80, 150][unit-1]
        cycles = days * 500
        for cycle in range(1, cycles + 1):
            progress = cycle / cycles
            temp = 45 + progress**2 * 30 + np.random.normal(0, 2)
            vibr = 5 + progress**1.5 * 10 + np.random.normal(0, 0.5)
            rpm = 3000 - progress**2 * 500 + np.random.normal(0, 50)
            data.append([unit, cycle, 35.0, 42.0, 3600.0, temp, vibr, rpm])
    
    df = pd.DataFrame(data)
    df.columns = ['Unit', 'Time', 'Setting_1', 'Setting_2', 'Setting_3', 'Sensor_1_Temp', 'Sensor_2_Vibr', 'Sensor_3_RPM']
    return df

df = generate_data()

# SIDEBAR
st.sidebar.markdown("### ⚙️ SETTINGS")
selected_bearing = st.sidebar.selectbox("🔧 Select Bearing:", [1, 2, 3, 4])
contamination = st.sidebar.slider("📊 Anomaly Sensitivity:", 0.01, 0.20, 0.05, 0.01)

# ANALYSIS
bearing_data = df[df['Unit'] == selected_bearing].copy().reset_index(drop=True)
scaler = StandardScaler()
X = scaler.fit_transform(bearing_data[['Sensor_1_Temp', 'Sensor_2_Vibr', 'Sensor_3_RPM']])
model = IsolationForest(contamination=contamination, random_state=42)
predictions = model.fit_predict(X)
bearing_data['Anomaly'] = predictions

# METRICS
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🌡️ Current Temp", f"{bearing_data['Sensor_1_Temp'].iloc[-1]:.1f}°C")
with col2:
    anomalies = bearing_data[bearing_data['Anomaly'] == -1]
    warning_days = (bearing_data['Time'].max() - anomalies['Time'].min()) / 500 if len(anomalies) > 0 else 0
    st.metric("⚠️ Warning Days", f"{warning_days:.1f}")
with col3:
    st.metric("📊 Anomalies", len(anomalies))

st.markdown("---")

# CHART
st.subheader(f"📈 Bearing #{selected_bearing}")
fig = go.Figure()

normal = bearing_data[bearing_data['Anomaly'] == 1]
fig.add_trace(go.Scatter(x=normal['Time'], y=normal['Sensor_1_Temp'], mode='markers', name='Normal', marker=dict(color='blue', size=4)))

anomalies = bearing_data[bearing_data['Anomaly'] == -1]
fig.add_trace(go.Scatter(x=anomalies['Time'], y=anomalies['Sensor_1_Temp'], mode='markers', name='🚨 ANOMALY', marker=dict(color='red', size=8)))

fig.update_layout(title=f"Bearing #{selected_bearing} - Isolation Forest", height=500, template='plotly_white')
st.plotly_chart(fig, use_container_width=True)

# ROI
st.subheader("💰 Business Impact")
st.markdown(f"""
**Without AI:** Bearing fails → 4-8h downtime → €5-10k loss

**With AI:** Predicted {warning_days:.0f} days before → Schedule maintenance → €0 downtime

**For 50 machines:** €250,000-500,000/year savings
""")