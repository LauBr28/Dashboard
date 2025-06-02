import streamlit as st
import pandas as pd
import joblib

# --- Carga de artefactos ---
preprocessor = joblib.load('preprocessor.pkl')
model        = joblib.load('final_productivity_model.pkl')

st.title("🚀 POC: Sistema de Predicción de Productividad Laboral")
st.markdown("""
**Solución basada en IA para identificar cómo los factores de estilo de vida y los hábitos digitales afectan el rendimiento laboral.**  
""")

with st.expander("📌 ¿Cómo funciona este modelo?", expanded=True):
    st.write("""
    Este modelo predictivo utiliza 19 variables de entrada (desde horas de sueño hasta uso de redes sociales) 
    para estimar el **`actual_productivity_score`** (0-10) con un **error promedio del 11.65% (MAPE)**.  
    🔍 **Hallazgos clave:**  
    - Variables críticas: Satisfacción laboral (20.3%), Horas de sueño (18.7%), Estrés (15.2%).  
    - Hábitos digitales explican solo el 9.8% del impacto.  
    """)
    st.image("feature_importance.png")  # Añade tu gráfico de importancia aquí

# --- Formulario de entrada ---

st.sidebar.header("⚙️ Configuración de Entradas")
st.sidebar.markdown("""*Complete los campos para simular diferentes perfiles de empleados.*""")
# (Mantén tu función user_input() actual)
def user_input():
    gender = st.sidebar.selectbox(
        "Género", ["Male", "Female", "Other"],
        help="Seleccione su género."
    )
    
    job_type = st.sidebar.selectbox(
        "Tipo de trabajo", ["IT", "Health", "Finance", "Student", "Unemployed"],
        help="Seleccione el tipo de trabajo que realiza actualmente."
    )
    
    platform = st.sidebar.selectbox(
        "Red social preferida", ["Instagram", "TikTok", "Other"],
        help="Red social que más utiliza."
    )

    data = {
        'number_of_notifications': st.sidebar.number_input(
            "Notificaciones por día (0–200)", min_value=0, max_value=200, value=60,
            help="Número promedio de notificaciones recibidas por día."
        ),
        'work_hours_per_day': st.sidebar.slider(
            "Horas de trabajo por día (0–16)", 0.0, 16.0, 8.0,
            help="Cantidad de horas que trabaja al día."
        ),
        'stress_level': st.sidebar.slider(
            "Nivel de estrés (1–10)", 1.0, 10.0, 5.0,
            help="Percepción personal del nivel de estrés diario."
        ),
        'sleep_hours': st.sidebar.slider(
            "Horas de sueño por noche (0–12)", 0.0, 12.0, 7.0,
            help="Promedio de horas que duerme cada noche."
        ),
        'screen_time_before_sleep': st.sidebar.slider(
            "Pantalla antes de dormir (0–5h)", 0.0, 5.0, 1.0,
            help="Tiempo que pasa frente a pantallas justo antes de dormir."
        ),
        'breaks_during_work': st.sidebar.number_input(
            "Pausas durante el trabajo (0–20)", min_value=0, max_value=20, value=5,
            help="Número promedio de pausas que realiza durante la jornada laboral."
        ),
        'uses_focus_apps': st.sidebar.selectbox(
            "¿Usa apps de enfoque? (1 = Sí, 0 = No)", [1, 0],
            help="¿Utiliza aplicaciones para mejorar su enfoque?"
        ),
        'has_digital_wellbeing_enabled': st.sidebar.selectbox(
            "¿Bienestar digital activado? (1 = Sí, 0 = No)", [1, 0],
            help="¿Tiene activadas funciones de bienestar digital en su teléfono?"
        ),
        'coffee_consumption_per_day': st.sidebar.number_input(
            "Tazas de café por día (0–10)", 0, 10, 2,
            help="Cantidad de café que consume diariamente."
        ),
        'days_feeling_burnout_per_month': st.sidebar.number_input(
            "Días con burnout al mes (0–31)", 0, 31, 5,
            help="Cantidad de días en los que se ha sentido agotado mentalmente en el último mes."
        ),
        'job_satisfaction_score': st.sidebar.slider(
            "Satisfacción laboral (0–10)", 0.0, 10.0, 5.0,
            help="Nivel de satisfacción con su trabajo actual."
        ),
        'social_media_log': st.sidebar.number_input(
            "Log del tiempo en redes (0.0–5.0)", 0.0, 5.0, 1.5,
            help="Logaritmo del tiempo diario estimado en redes sociales."
        ),
        'gender': gender,
        'job_type': job_type,
        'social_platform_preference': platform
    }
    return pd.DataFrame([data])



df_input = user_input()

# --- Preprocesar y predecir ---
st.subheader("Datos procesados")
st.write(df_input)

X_proc = preprocessor.transform(df_input)


pred   = model.predict(X_proc)

st.subheader("📊 Resultados de la Predicción")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="**Productividad Predicha** (0-10)", value=f"{pred[0]:.2f}", 
              delta=f"{'↑ Alta' if pred[0]>=7.5 else ('→ Media' if pred[0]>=5 else '↓ Baja')}")
with col2:
    st.metric(label="**Error Estimado**", value="±1.13 puntos", 
              help="RMSE del modelo en datos de prueba")

# Barra de progreso interpretativa
st.progress(pred[0]/10)
st.caption(f"**Interpretación:** {nivel} productividad ({'Se recomiendan intervenciones' if pred[0]<5 else 'Rendimiento adecuado' if pred[0]<7.5 else 'Desempeño óptimo'})")


st.subheader("📈 Simulador de Escenarios")
variable = st.selectbox("¿Qué variable desea simular?", 
                       ["sleep_hours", "social_media_log", "job_satisfaction_score"])
range_values = st.slider(f"Rango de {variable}", 0.0, 10.0 if variable!="sleep_hours" else 12.0, (3.0, 7.0))

# Simulación básica (ejemplo)
sim_values = [range_values[0], (range_values[0]+range_values[1])/2, range_values[1]]
sim_results = []
for val in sim_values:
    df_sim = df_input.copy()
    df_sim[variable] = val
    sim_results.append(model.predict(preprocessor.transform(df_sim))[0])

sim_data = pd.DataFrame({
    variable: sim_values,
    "Productividad": sim_results
})

st.line_chart(sim_data.set_index(variable))

st.subheader("🎯 Recomendaciones Personalizadas")
if pred[0] < 5:
    st.warning("""
    ⚠️ **Acciones prioritarias:**  
    - Mejorar satisfacción laboral (actual: {:.1f}/10).  
    - Aumentar horas de sueño (actual: {:.1f}h).  
    - Reducir estrés (actual: {:.1f}/10).  
    - Limitar tiempo en redes sociales (actual: {:.1f} log-horas).  
    """.format(df_input['job_satisfaction_score'].iloc[0], 
              df_input['sleep_hours'].iloc[0],
              df_input['stress_level'].iloc[0],
              df_input['social_media_log'].iloc[0]))
elif pred[0] < 7.5:
    st.info("""
    ℹ️ **Oportunidades de mejora:**  
    - Optimizar pausas laborales (actuales: {} por día).  
    - Reducir pantallas antes de dormir (actual: {:.1f}h).  
    - Monitorear días de burnout (actuales: {} mensuales).  
    """.format(df_input['breaks_during_work'].iloc[0],
              df_input['screen_time_before_sleep'].iloc[0],
              df_input['days_feeling_burnout_per_month'].iloc[0]))
else:
    st.success("✅ **Rendimiento óptimo:** Mantenga sus hábitos actuales.")


st.markdown("---")
st.markdown("""
**🔧 Tecnología:**  
- Modelo: Gradient Boosting (R²=0.65, RMSE=1.13).  
- Datos: Simulados (30k registros) con 19 características.  
""")