# cnc-predictive-maintenance
# 🔧CNC Bearing Predictive Maintenance AI

Sistem AI do przewidywania awarii łożysk w maszynach CNC. 
**Ostrzega 100+ dni PRZED awarią.**

## Problem
- Łożysko się psuje niespodziewanie → 4-8h downtime → €5-10k strata
- Brak wskaźnika kiedy wymienić bearing

## Rozwiązanie
- Model ML (Isolation Forest) szuka anomalii w sensorach
- Streamlit Dashboard do wizualizacji
- FastAPI Backend do predykcji
- **Rezultat: ostrzeżenie 100+ dni wcześniej**

## Business Impact
| Scenariusz | Koszt |
|-----------|-------|
| Bez AI | 4-8h downtime → €5-10k strata |
| Z AI | Planowa wymiana → €0 strata |
| **Na 50 maszyn** | **€250,000-500,000/rok** |

---

## 🚀 Jak uruchomić

### Wymagania
- Python 3.8+
- pip

### Instalacja
```bash
git clone https://github.com/michaldob21/cnc-predictive-maintenance
cd cnc-predictive-maintenance
pip install -r requirements.txt
```

### Uruchomienie
**Frontend (Streamlit Dashboard):**
```bash
streamlit run app.py
```
Potem: http://localhost:8501

**Backend (FastAPI):**
```bash
python main.py
```
Potem: http://localhost:8000/docs (Swagger UI)

---

## Co robi aplikacja?

### Frontend (app.py)
- Interaktywny dashboard Streamlit
- Wybór łożyska (dropdown)
- Slider do zmiany czułości detektora
- 3 sensory (Temperatura, Wibracje, RPM)
- Progress bar (% życia bearing'u)
- Info o maszynie (czas pracy, cykle, średnia temp)

### Backend (main.py)
- REST API (FastAPI)
- `/` - Info endpoint
- `/health` - Health check
- `/predict` - Predykcja anomalii
- Swagger dokumentacja (/docs)

---

## Technologia

| Warstwa | Narzędzie | Czemu |
|---------|-----------|-------|
| **Data Processing** | Pandas, NumPy | Manipulacja 225k wierszy danych |
| **ML Model** | scikit-learn (Isolation Forest) | Anomaly detection bez labelów |
| **Frontend** | Streamlit | Dashboard bez JavaScript |
| **Backend** | FastAPI | REST API z auto-dokumentacją |
| **Deployment** | Streamlit Cloud + Replit | Free, auto-deploy |
| **Version Control** | GitHub | Public repository |

---

## Jak działa (uproszczenie)

1. **Generowanie danych**
   - 4 bearing'i × 100-150 dni pracy
   - 225,000 pomiarów (temperatura, wibracje, RPM)

2. **Trenowanie modelu**
   - Isolation Forest szuka outliers
   - Parametr: `contamination=0.05` (szukaj top 5% anomalii)

3. **Predykcja**
   - Gdzie zaczęły się anomalie?
   - Ile dni do awarii?
   - Status (OK / ALERT)

4. **Wizualizacja**
   - Dashboard pokazuje wykresy
   - Progress bar (% życia)
   - Metryki (temperatura, anomalie, dni)


---

## Struktura
