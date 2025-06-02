import streamlit as st
import pandas as pd
import joblib

# --- Carga de artefactos ---
preprocessor = joblib.load('preprocessor.pkl')
model        = joblib.load('final_productivity_model.pkl')

# --- Título y descripción ---
st.title("🚀 POC: Sistema de Predicción de Productividad Laboral")
st.markdown("""
*Solución basada en IA para identificar cómo los factores de estilo de vida y los hábitos digitales afectan el rendimiento laboral.*  
""")

with st.expander("📌 ¿Cómo funciona este modelo?", expanded=True):
    st.write("""
    Este modelo predictivo utiliza 19 variables de entrada (desde horas de sueño hasta uso de redes sociales) 
    para estimar el *actual_productivity_score* (0-10) con un *error promedio del 11.65% (MAPE)*.  
    🔍 *Hallazgos clave:*  
    - Variables críticas: Satisfacción laboral (20.3%), Horas de sueño (18.7%), Estrés (15.2%).  
    - Hábitos digitales explican solo el 9.8% del impacto.  
    """)

# --- Formulario de entrada ---
st.sidebar.header("⚙️ Configuración de Entradas")
st.sidebar.markdown("""Complete los campos para simular diferentes perfiles de empleados.""")

def user_input():
    gender = st.sidebar.selectbox("Género", ["Male", "Female", "Other"])
    job_type = st.sidebar.selectbox("Tipo de trabajo", ["IT", "Health", "Finance", "Student", "Unemployed"])
    platform = st.sidebar.selectbox("Red social preferida", ["Instagram", "TikTok", "Other"])

    data = {
        'number_of_notifications': st.sidebar.number_input("Notificaciones por día (0–200)", 0, 200, 60),
        'work_hours_per_day': st.sidebar.slider("Horas de trabajo por día (0–16)", 0.0, 16.0, 8.0),
        'stress_level': st.sidebar.slider("Nivel de estrés (1–10)", 1.0, 10.0, 5.0),
        'sleep_hours': st.sidebar.slider("Horas de sueño por noche (0–12)", 0.0, 12.0, 7.0),
        'screen_time_before_sleep': st.sidebar.slider("Pantalla antes de dormir (0–5h)", 0.0, 5.0, 1.0),
        'breaks_during_work': st.sidebar.number_input("Pausas durante el trabajo (0–20)", 0, 20, 5),
        'uses_focus_apps': st.sidebar.selectbox("¿Usa apps de enfoque? (1 = Sí, 0 = No)", [1, 0]),
        'has_digital_wellbeing_enabled': st.sidebar.selectbox("¿Bienestar digital activado? (1 = Sí, 0 = No)", [1, 0]),
        'coffee_consumption_per_day': st.sidebar.number_input("Tazas de café por día (0–10)", 0, 10, 2),
        'days_feeling_burnout_per_month': st.sidebar.number_input("Días con burnout al mes (0–31)", 0, 31, 5),
        'job_satisfaction_score': st.sidebar.slider("Satisfacción laboral (0–10)", 0.0, 10.0, 5.0),
        'social_media_log': st.sidebar.number_input("Log del tiempo en redes (0.0–5.0)", 0.0, 5.0, 1.5),
        'gender': gender,
        'job_type': job_type,
        'social_platform_preference': platform
    }
    return pd.DataFrame([data])

# --- Recogida de entrada ---
df_input = user_input()

# --- Visualizar entrada ---
st.subheader("📥 Datos procesados")
st.write(df_input)

# --- Predecir ---
X_proc = preprocessor.transform(df_input)
pred = model.predict(X_proc)

# --- Interpretación ---
nivel = "Alta" if pred[0] >= 7.5 else ("Media" if pred[0] >= 5 else "Baja")

# --- Resultados ---
st.subheader("📊 Resultados de la Predicción")
col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="*Productividad Predicha* (0-10)", 
        value=f"{pred[0]:.2f}", 
        delta=f"{'↑ Alta' if pred[0]>=7.5 else ('→ Media' if pred[0]>=5 else '↓ Baja')}"
    )
with col2:
    st.metric(
        label="*Error Estimado*", 
        value="±1.13 puntos", 
        help="RMSE del modelo en datos de prueba"
    )

# --- Barra de progreso ---
st.progress(pred[0]/10)
st.caption(f"*Interpretación:* {nivel} productividad ({'Se recomiendan intervenciones' if pred[0]<5 else 'Rendimiento adecuado' if pred[0]<7.5 else 'Desempeño óptimo'})")

# --- Recomendaciones personalizadas ---
st.subheader("🎯 Recomendaciones Personalizadas")
if pred[0] < 5:
    st.warning(f"""
⚠️ *Acciones prioritarias:*  
- Mejorar satisfacción laboral (actual: {df_input['job_satisfaction_score'].iloc[0]:.1f}/10).  
- Aumentar horas de sueño (actual: {df_input['sleep_hours'].iloc[0]:.1f}h).  
- Reducir estrés (actual: {df_input['stress_level'].iloc[0]:.1f}/10).  
- Limitar tiempo en redes sociales (actual: {df_input['social_media_log'].iloc[0]:.1f} log-horas).  
    """)
elif pred[0] < 7.5:
    st.info(f"""
ℹ️ *Oportunidades de mejora:*  
- Optimizar pausas laborales (actuales: {df_input['breaks_during_work'].iloc[0]} por día).  
- Reducir pantallas antes de dormir (actual: {df_input['screen_time_before_sleep'].iloc[0]:.1f}h).  
- Monitorear días de burnout (actuales: {df_input['days_feeling_burnout_per_month'].iloc[0]} mensuales).  
    """)
else:
    st.success("✅ *Rendimiento óptimo:* Mantenga sus hábitos actuales.")

# --- Información adicional ---
st.markdown("---")
st.markdown("""
*🔧 Tecnología:*  
- Modelo: Gradient Boosting (R²=0.65, RMSE=1.13).  
- Datos: 30k registros con 19 características.  
""")
