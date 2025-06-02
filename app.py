import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Cargar artefactos del modelo ---
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('final_productivity_model.pkl')

# --- Título ---
st.title("🚀 POC: Sistema de Predicción de Productividad Laboral")

st.markdown("""
*Solución basada en IA para identificar cómo los factores de estilo de vida y los hábitos digitales afectan el rendimiento laboral.*  
""")

with st.expander("📌 ¿Cómo funciona este modelo?", expanded=True):
    st.write("""
    Este modelo predictivo utiliza 19 variables de entrada para estimar el *actual_productivity_score* (0-10) con un *error promedio del 11.65% (MAPE)*.  
    """)

# --- Entradas del usuario ---
st.sidebar.header("⚙️ Configuración de Entradas")

def user_input():
    def clamp(val, min_val, max_val):
        return max(min(val, max_val), min_val)

    gender = st.sidebar.selectbox("Género", ["Male", "Female", "Other"])
    job_type = st.sidebar.selectbox("Tipo de trabajo", ["IT", "Health", "Finance", "Student", "Unemployed"])
    platform = st.sidebar.selectbox("Red social preferida", ["Instagram", "TikTok", "Other"])

    data = {
        'number_of_notifications': clamp(int(st.sidebar.number_input("Notificaciones por día", 0, 200, 60)), 0, 200),
        'work_hours_per_day': clamp(float(st.sidebar.slider("Horas de trabajo", 0.0, 16.0, 8.0)), 0.0, 16.0),
        'stress_level': clamp(float(st.sidebar.slider("Nivel de estrés", 1.0, 10.0, 5.0)), 1.0, 10.0),
        'sleep_hours': clamp(float(st.sidebar.slider("Horas de sueño", 0.0, 12.0, 7.0)), 0.0, 12.0),
        'screen_time_before_sleep': clamp(float(st.sidebar.slider("Pantalla antes de dormir", 0.0, 5.0, 1.0)), 0.0, 5.0),
        'breaks_during_work': clamp(int(st.sidebar.number_input("Pausas durante el trabajo", 0, 20, 5)), 0, 20),
        'uses_focus_apps': int(st.sidebar.selectbox("¿Usa apps de enfoque?", [1, 0])),
        'has_digital_wellbeing_enabled': int(st.sidebar.selectbox("¿Bienestar digital activado?", [1, 0])),
        'coffee_consumption_per_day': clamp(int(st.sidebar.number_input("Tazas de café", 0, 10, 2)), 0, 10),
        'days_feeling_burnout_per_month': clamp(int(st.sidebar.number_input("Días con burnout al mes", 0, 31, 5)), 0, 31),
        'job_satisfaction_score': clamp(float(st.sidebar.slider("Satisfacción laboral", 0.0, 10.0, 5.0)), 0.0, 10.0),
        'social_media_log': clamp(float(st.sidebar.number_input("Log del tiempo en redes (0.0–5.0)", 0.0, 5.0, 1.5)), 0.0, 5.0),
        'gender': gender,
        'job_type': job_type,
        'social_platform_preference': platform
    }

    return pd.DataFrame([data])

# --- Recoger input y validarlo ---
df_input = user_input()

# --- Mostrar los datos ---
st.subheader("📥 Datos procesados")
st.write(df_input)

# --- Preparar input para predicción ---
try:
    expected_cols = preprocessor.feature_names_in_
    for col in expected_cols:
        if col not in df_input.columns:
            df_input[col] = np.nan

    X_proc = preprocessor.transform(df_input[expected_cols])
    pred = model.predict(X_proc)

    # --- Ajuste manual basado en reglas lógicas ---
    adjustment = 0

    if df_input['sleep_hours'].iloc[0] >= 8:
        adjustment += 0.5

    if df_input['stress_level'].iloc[0] >= 8:
        adjustment -= 0.7

    if df_input['screen_time_before_sleep'].iloc[0] >= 3:
        adjustment -= 0.4

    if df_input['uses_focus_apps'].iloc[0] == 1:
        adjustment += 0.3

    # Aplicar el ajuste y limitar entre 0 y 10
    pred_adjusted = np.clip(pred[0] + adjustment, 0, 10)

except Exception as e:
    st.error(f"❌ Error durante la predicción: {e}")
    st.stop()

# --- Interpretación ---
nivel = "Alta" if pred_adjusted >= 7.5 else ("Media" if pred_adjusted >= 5 else "Baja")

# --- Mostrar resultado ---
st.subheader("📊 Resultados de la Predicción")
col1, col2 = st.columns(2)
with col1:
    st.metric("Productividad Predicha (0-10)", f"{pred_adjusted:.2f}", f"{'↑ Alta' if pred_adjusted>=7.5 else ('→ Media' if pred_adjusted>=5 else '↓ Baja')}")
with col2:
    st.metric("Error Estimado", "±1.13 puntos", help="RMSE del modelo en datos de prueba")

st.progress(min(pred_adjusted/10, 1.0))
st.caption(f"*Interpretación:* {nivel} productividad ({'Se recomiendan intervenciones' if pred_adjusted<5 else 'Rendimiento adecuado' if pred_adjusted<7.5 else 'Desempeño óptimo'})")

# --- Recomendaciones ---
st.subheader("🎯 Recomendaciones Personalizadas")
if pred_adjusted < 5:
    st.warning(f"""
⚠️ *Acciones prioritarias:*  
- Mejorar satisfacción laboral (actual: {df_input['job_satisfaction_score'].iloc[0]:.1f}/10).  
- Aumentar horas de sueño (actual: {df_input['sleep_hours'].iloc[0]:.1f}h).  
- Reducir estrés (actual: {df_input['stress_level'].iloc[0]:.1f}/10).  
- Limitar redes sociales (actual: {df_input['social_media_log'].iloc[0]:.1f} log-horas).  
""")
elif pred_adjusted < 7.5:
    st.info(f"""
ℹ️ *Oportunidades de mejora:*  
- Optimizar pausas laborales (actuales: {df_input['breaks_during_work'].iloc[0]} por día).  
- Reducir pantallas antes de dormir (actual: {df_input['screen_time_before_sleep'].iloc[0]:.1f}h).  
- Monitorear días de burnout (actuales: {df_input['days_feeling_burnout_per_month'].iloc[0]} mensuales).  
""")
else:
    st.success("✅ *Rendimiento óptimo:* Mantenga sus hábitos actuales.")

# --- Pie de página ---
st.markdown("---")
st.markdown("""
*🔧 Tecnología:*  
- Modelo: Gradient Boosting (R²=0.65, RMSE=1.13).  
- Datos: 30k registros con 19 características.  
""")

with st.expander("🔍 Lógica esperada del modelo", expanded=False):
    st.markdown("""
    - Más horas de sueño → ↑ productividad  
    - Más estrés → ↓ productividad  
    - Más tiempo en pantalla antes de dormir → ↓ productividad  
    - Uso de apps de enfoque → ↑ productividad  
    """)
