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

st.subheader("Predicción de Productividad")
st.metric(label="Score estimado (0–10)", value=f"{pred[0]:.2f}")

# --- Interpretación sencilla ---
nivel = "Alta" if pred[0]>=7.5 else ("Media" if pred[0]>=5 else "Baja")
st.write(f"▶️ **Nivel:** {nivel} productividad")
