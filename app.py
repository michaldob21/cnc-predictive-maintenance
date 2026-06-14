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
        days = [100, 120, 80, 150][unit - 1]
        cycles = days * 500
        for cycle in range(1, cycles + 1):
            progress = cycle / cycles
            temp = 45 + progress**2 * 30 + np.random.normal(0, 2)
            vibr = 5 + progress**1.5 * 10 + np.random.normal(0, 0.5)
            rpm = 3000 - progress**2 * 500 + np.random.normal(0, 50)
            data.append([unit, cycle, 35.0, 42.0, 3600.0, temp, vibr, rpm])

    df = pd.DataFrame(data)
    df.columns = [
        "Unit",
        "Time",
        "Setting_1",
        "Setting_2",
        "Setting_3",
        "Sensor_1_Temp",
        "Sensor_2_Vibr",
        "Sensor_3_RPM",
    ]
    return df


df = generate_data()

# SIDEBAR
st.sidebar.markdown("### ⚙️ SETTINGS")
bearing_names = {
    1: "Łożysko Wrzeciona Głównego",
    2: "Łożysko Osi X",
    3: "Łożysko Osi Z",
    4: "Łożysko Pompy Chłodzenia",
}

bearing_list = list(bearing_names.values())
selected_bearing_name = st.sidebar.selectbox("🔧 Wybierz Łożysko:", bearing_list)
selected_bearing = list(bearing_names.keys())[bearing_list.index(selected_bearing_name)]
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 Jak działa Czułość?

- **Niska (0.01-0.05)**: Szukaj TYLKO rzeczywistych anomalii
  - Mniej alertów
  - Ostrzeżenie wcześniej

- **Wysoka (0.10-0.20)**: Szukaj "podejrzanych" punktów
  - Więcej alertów
  - Ostrzeżenie później
""")
sensitivity = st.sidebar.slider(
    "🎚️ Czułość Detektora:",
    0.01,
    0.20,
    0.05,
    0.01,
)
contamination = sensitivity


# ANALYSIS
bearing_data = df[df["Unit"] == selected_bearing].copy().reset_index(drop=True)
scaler = StandardScaler()
X = scaler.fit_transform(
    bearing_data[["Sensor_1_Temp", "Sensor_2_Vibr", "Sensor_3_RPM"]]
)
model = IsolationForest(contamination=contamination, random_state=42)
predictions = model.fit_predict(X)
bearing_data["Anomaly"] = predictions
# MACHINE INFO
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏭 Info o Maszynie")
total_days = bearing_data["Time"].max() / 500
total_cycles = bearing_data["Time"].max()
st.sidebar.metric("⏱️ Całkowity Czas Pracy", f"{int(total_days)} dni")
st.sidebar.metric("🔄 Liczba Cykli", f"{int(total_cycles):,}")
avg_temp = bearing_data["Sensor_1_Temp"].mean()
st.sidebar.metric("🌡️ Średnia Temp.", f"{avg_temp:.1f}°C")
# METRICS
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🌡️ Current Temp", f"{bearing_data['Sensor_1_Temp'].iloc[-1]:.1f}°C")
with col2:
    anomalies = bearing_data[bearing_data["Anomaly"] == -1]
    warning_days = (
        (bearing_data["Time"].max() - anomalies["Time"].min()) / 500
        if len(anomalies) > 0
        else 0
    )
    st.metric("⚠️ Warning Days", f"{warning_days:.1f}")
with col3:
    st.metric("📊 Anomalies", len(anomalies))
st.markdown("---")

# PROGRESS BAR
anomalies = bearing_data[bearing_data["Anomaly"] == -1]
total_life_cycles = bearing_data["Time"].max()

if len(anomalies) > 0:
    degradation_start = anomalies["Time"].min()
    cycles_left = total_life_cycles - degradation_start
    progress_percent = (cycles_left / total_life_cycles) * 100
else:
    progress_percent = 100

st.write(f"**Pozostałe życie: {progress_percent:.0f}%**")
st.progress(progress_percent / 100)

if progress_percent > 30:
    st.success("✅ Bearing w dobrej kondycji")
elif progress_percent > 10:
    st.warning("⚠️ Planuj wymianę")
else:
    st.error("🔴 WYMIEŃ SZYBKO!")
st.markdown("---")

# CHART
st.subheader(f"📈 {bearing_names[selected_bearing]}")

# SENSOR TOGGLE
sensor_choice = st.radio(
    "Wybierz Sensor:", ["🌡️ Temperatura", "📈 Wibracje", "🔄 RPM"], horizontal=True
)

fig = go.Figure()

if sensor_choice == "🌡️ Temperatura":
    normal = bearing_data[bearing_data["Anomaly"] == 1]
    fig.add_trace(
        go.Scatter(
            x=normal["Time"],
            y=normal["Sensor_1_Temp"],
            mode="markers",
            name="Normal",
            marker=dict(color="blue", size=5),
        )
    )

    anomalies = bearing_data[bearing_data["Anomaly"] == -1]
    fig.add_trace(
        go.Scatter(
            x=anomalies["Time"],
            y=anomalies["Sensor_1_Temp"],
            mode="markers",
            name="🚨 Anomaly",
            marker=dict(color="red", size=8),
        )
    )

    fig.update_layout(
        title=f"{bearing_names[selected_bearing]} - Temperatura",
        xaxis_title="Cykl produkcji",
        yaxis_title="Temperatura (°C)",
        height=500,
        template="plotly_dark",
    )

elif sensor_choice == "📈 Wibracje":
    normal = bearing_data[bearing_data["Anomaly"] == 1]
    fig.add_trace(
        go.Scatter(
            x=normal["Time"],
            y=normal["Sensor_2_Vibr"],
            mode="lines",
            name="Normal",
            line=dict(color="cyan", width=2),
        )
    )

    anomalies = bearing_data[bearing_data["Anomaly"] == -1]
    fig.add_trace(
        go.Scatter(
            x=anomalies["Time"],
            y=anomalies["Sensor_2_Vibr"],
            mode="lines",
            name="🚨 Anomaly",
            line=dict(color="orange", width=3),
        )
    )

    fig.update_layout(
        title=f"{bearing_names[selected_bearing]} - Wibracje",
        xaxis_title="Cykl produkcji",
        yaxis_title="Wibracje (mm/s)",
        height=500,
        template="plotly_dark",
    )

elif sensor_choice == "🔄 RPM":
    normal = bearing_data[bearing_data["Anomaly"] == 1]
    fig.add_trace(
        go.Scatter(
            x=normal["Time"],
            y=normal["Sensor_3_RPM"],
            mode="lines",
            name="Normal",
            line=dict(color="green", width=2),
        )
    )

    anomalies = bearing_data[bearing_data["Anomaly"] == -1]
    fig.add_trace(
        go.Scatter(
            x=anomalies["Time"],
            y=anomalies["Sensor_3_RPM"],
            mode="lines",
            name="🚨 Anomaly",
            line=dict(color="red", width=3),
        )
    )

    fig.update_layout(
        title=f"{bearing_names[selected_bearing]} - RPM",
        xaxis_title="Cykl produkcji",
        yaxis_title="RPM",
        height=500,
        template="plotly_dark",
    )

st.plotly_chart(fig, use_container_width=True)

# ROI
st.subheader("💰 Business Impact")
st.markdown(f"""
**Without AI:** Bearing fails → 4-8h downtime → €5-10k loss

**With AI:** Predicted {warning_days:.0f} days before → Schedule maintenance → €0 downtime

**For 50 machines:** €250,000-500,000/year savings
""")
