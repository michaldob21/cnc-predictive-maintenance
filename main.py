from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

bearing_names = {
    1: "Łożysko Wrzeciona Głównego",
    2: "Łożysko Osi X",
    3: "Łożysko Osi Z",
    4: "Łożysko Pompy Chłodzenia"
}

app = FastAPI(
    title="CNC Bearing AI API",
    description="Predictive maintenance API",
    version="1.0"
)

# CORS (pozwala Streamlit się łączyć)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "CNC Bearing Predictive Maintenance API",
        "version": "1.0",
        "endpoints": ["/docs", "/health", "/predict"]
    }

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/predict")
def predict(bearing_unit: int = 1, contamination: float = 0.05):
    """
    Predict anomalies for a bearing
    bearing_unit: 1-4
    contamination: 0.01-0.20
    """
    
    np.random.seed(42)
    data = []
    days = [100, 120, 80, 150][bearing_unit-1]
    cycles = days * 500
    
    for cycle in range(1, cycles + 1):
        progress = cycle / cycles
        temp = 45 + progress**2 * 30 + np.random.normal(0, 2)
        vibr = 5 + progress**1.5 * 10 + np.random.normal(0, 0.5)
        rpm = 3000 - progress**2 * 500 + np.random.normal(0, 50)
        data.append([bearing_unit, cycle, 35.0, 42.0, 3600.0, temp, vibr, rpm])
    
    df = pd.DataFrame(data)
    df.columns = ['Unit', 'Time', 'S1', 'S2', 'S3', 'Temp', 'Vibr', 'RPM']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['Temp', 'Vibr', 'RPM']])
    
    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(X)
    
    anomalies_idx = np.where(predictions == -1)[0]
    
    if len(anomalies_idx) > 0:
        warning_days = (len(df) - anomalies_idx.min()) / 500
    else:
        warning_days = 0
    
    return {
    "bearing_id": bearing_unit,
    "bearing_name": bearing_names[bearing_unit],
    "total_cycles": len(df),
    "anomalies_found": int(len(anomalies_idx)),
    "warning_days": float(warning_days),
    "status": "OK" if warning_days > 30 else "ALERT",
    "current_temp": float(df['Temp'].iloc[-1])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
