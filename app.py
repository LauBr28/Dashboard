import streamlit as st
import pandas as pd
import joblib

# --- Carga de artefactos ---
preprocessor = joblib.load('preprocessor.pkl')
model        = joblib.load('final_productivity_model.pkl')

st.title("POC: Predicción de Productividad 📊")

st.markdown("""
Este dashboard muestra cómo, a partir de datos de hábitos digitales,
nuestro modelo predice la productividad real (0–10).
""")

# --- Formulario de entrada ---
st.sidebar.header("Input de usuario")
def user_input():
    data = {
      'number_of_notifications': st.sidebar.number_input("Notificaciones/día", min_value=0, max_value=200, value=60),
      'work_hours_per_day':      st.sidebar.slider("Horas de trabajo/día", 0.0, 16.0, 8.0),
      'stress_level':            st.sidebar.slider("Estrés (1–10)", 1.0, 10.0, 5.0),
      'sleep_hours':             st.sidebar.slider("Sueño (h/noches)", 0.0, 12.0, 7.0),
      'screen_time_before_sleep':st.sidebar.slider("Tiempo pantalla antes de dormir", 0.0, 5.0, 1.0),
      'breaks_during_work':      st.sidebar.number_input("Pausas en trabajo", min_value=0, max_value=20, value=5),
      'uses_focus_apps':         st.sidebar.selectbox("Usa apps de enfoque", [0,1]),
      'has_digital_wellbeing_enabled': st.sidebar.selectbox("Bienestar digital ON", [0,1]),
      'coffee_consumption_per_day':     st.sidebar.number_input("Tazas de café/día", 0, 10, 2),
      'days_feeling_burnout_per_month': st.sidebar.number_input("Días de burnout/mes", 0, 31, 5),
      'job_satisfaction_score':         st.sidebar.slider("Satisfacción (0–10)", 0.0, 10.0, 5.0),
      'social_media_log':               st.sidebar.number_input("Log(SM time)", 0.0, 5.0, 1.5),
      # Para variables one-hot, usa selectbox y luego convertir a 0/1:
      'gender_Male':      int(st.sidebar.selectbox("Género", ["Other","Male","Female"])=="Male"),
      'gender_Other':     int(st.sidebar.selectbox("", ["Other","Male","Female"])=="Other"),
      # ... y así con job_type_* y social_platform_preference_*
    }
    return pd.DataFrame([data])

df_input = user_input()

# --- Preprocesar y predecir ---
st.subheader("Datos procesados")
st.write(df_input)

X_proc = preprocessor.transform(df_input)
pred   = model.predict(X_proc)

st.subheader("Predicción de Productividad")
st.metric(label="Score estimado (0–10)", value=f"{pred[0]:.2f}")

# --- Interpretación sencilla ---
nivel = "Alta" if pred[0]>=7.5 else ("Media" if pred[0]>=5 else "Baja")
st.write(f"▶️ **Nivel:** {nivel} productividad")
