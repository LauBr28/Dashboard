import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --- Carga de artefactos ---
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('final_productivity_model.pkl')

# Definir importancia de características (según tu gráfico)
feature_importance = {
    'job_satisfaction_score': 20.3,
    'sleep_hours': 18.7,
    'stress_level': 15.2,
    'social_media_log': 9.8,
    'work_hours_per_day': 7.5,
    'screen_time_before_sleep': 4.1,
    'days_feeling_burnout_per_month': 3.9,
    'number_of_notifications': 3.5,
    'job_type_Unemployed': 2.8,
    'breaks_during_work': 2.5,
    'coffee_consumption_per_day': 2.3,
    'job_type_Student': 1.9,
    'uses_focus_apps': 1.7,
    'job_type_Finance': 1.5,
    'social_platform_preference_Instagram': 1.2,
    'has_digital_wellbeing_enabled': 0.9,
    'gender_Male': 0.7,
    'social_platform_preference_Other': 0.5,
    'job_type_IT': 0.4,
    'job_type_Health': 0.3,
    'gender_Other': 0.2,
    'social_platform_preference_TikTok': 0.1
}

# --- Configuración de la página ---
st.set_page_config(page_title="Predictor de Productividad", layout="wide")

# --- Header ---
st.title("🚀 POC: Sistema de Predicción de Productividad Laboral")
st.markdown("""
**Solución basada en IA para identificar cómo los factores de estilo de vida y los hábitos digitales afectan el rendimiento laboral.**  
""")

with st.expander("📌 ¿Cómo funciona este modelo?", expanded=True):
    st.write("""
    Este modelo predictivo utiliza 19 variables de entrada para estimar el **`actual_productivity_score`** (0-10) con un **error promedio del 11.65% (MAPE)**.  
    🔍 **Hallazgos clave:**  
    - Variables críticas: Satisfacción laboral (20.3%), Horas de sueño (18.7%), Estrés (15.2%)  
    - Hábitos digitales explican solo el 9.8% del impacto  
    """)
    
    # Gráfico de importancia de características
    fig, ax = plt.subplots(figsize=(10, 6))
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    ax.barh(features[:5][::-1], importance[:5][::-1], color='#1f77b4')
    ax.barh(features[5:][::-1], importance[5:][::-1], color='#aec7e8')
    ax.set_xlabel('Importancia Relativa (%)')
    ax.set_title('Top 5 Características Más Importantes (71.5% impacto total)')
    st.pyplot(fig)

# --- Sidebar con inputs ---
st.sidebar.header("⚙️ Configuración de Entradas")
st.sidebar.markdown("""*Complete los campos para simular diferentes perfiles de empleados.*""")

def user_input():
    data = {
        'job_satisfaction_score': st.sidebar.slider(
            "Satisfacción laboral (0-10)", 0.0, 10.0, 5.0,
            help=f"Impacto: {feature_importance['job_satisfaction_score']}%"
        ),
        'sleep_hours': st.sidebar.slider(
            "Horas de sueño por noche (0-12)", 0.0, 12.0, 7.0,
            help=f"Impacto: {feature_importance['sleep_hours']}%"
        ),
        'stress_level': st.sidebar.slider(
            "Nivel de estrés (1-10)", 1.0, 10.0, 5.0,
            help=f"Impacto: {feature_importance['stress_level']}%"
        ),
        'social_media_log': st.sidebar.number_input(
            "Log del tiempo en redes (0.0-5.0)", 0.0, 5.0, 1.5,
            help=f"Impacto: {feature_importance['social_media_log']}%"
        ),
        'work_hours_per_day': st.sidebar.slider(
            "Horas de trabajo por día (0-16)", 0.0, 16.0, 8.0,
            help=f"Impacto: {feature_importance['work_hours_per_day']}%"
        ),
        'screen_time_before_sleep': st.sidebar.slider(
            "Pantalla antes de dormir (0-5h)", 0.0, 5.0, 1.0,
            help=f"Impacto: {feature_importance['screen_time_before_sleep']}%"
        ),
        'days_feeling_burnout_per_month': st.sidebar.number_input(
            "Días con burnout al mes (0-31)", 0, 31, 5,
            help=f"Impacto: {feature_importance['days_feeling_burnout_per_month']}%"
        ),
        'number_of_notifications': st.sidebar.number_input(
            "Notificaciones por día (0-200)", min_value=0, max_value=200, value=60,
            help=f"Impacto: {feature_importance['number_of_notifications']}%"
        ),
        'gender': st.sidebar.selectbox(
            "Género", ["Male", "Female", "Other"],
            help=f"Impacto: {feature_importance['gender_Male']}%"
        ),
        'job_type': st.sidebar.selectbox(
            "Tipo de trabajo", ["IT", "Health", "Finance", "Student", "Unemployed"],
            help="Seleccione el tipo de trabajo"
        ),
        'social_platform_preference': st.sidebar.selectbox(
            "Red social preferida", ["Instagram", "TikTok", "Other"],
            help="Red social que más utiliza"
        ),
        'breaks_during_work': st.sidebar.number_input(
            "Pausas durante el trabajo (0-20)", min_value=0, max_value=20, value=5,
            help=f"Impacto: {feature_importance['breaks_during_work']}%"
        ),
        'uses_focus_apps': st.sidebar.selectbox(
            "¿Usa apps de enfoque?", [1, 0],
            help=f"Impacto: {feature_importance['uses_focus_apps']}%"
        ),
        'has_digital_wellbeing_enabled': st.sidebar.selectbox(
            "¿Bienestar digital activado?", [1, 0],
            help=f"Impacto: {feature_importance['has_digital_wellbeing_enabled']}%"
        ),
        'coffee_consumption_per_day': st.sidebar.number_input(
            "Tazas de café por día (0-10)", 0, 10, 2,
            help=f"Impacto: {feature_importance['coffee_consumption_per_day']}%"
        )
    }
    return pd.DataFrame([data])

df_input = user_input()

# --- Preprocesar y predecir ---
st.subheader("📋 Datos de Entrada")
st.dataframe(df_input.style.highlight_max(axis=0, color='#d4f1f9'), height=150)

X_proc = preprocessor.transform(df_input)
pred = model.predict(X_proc)

# --- Resultados ---
st.subheader("📊 Resultados de la Predicción")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Productividad Predicha (0-10)", f"{pred[0]:.2f}")
with col2:
    st.metric("Nivel de Productividad", 
              "Alta" if pred[0] >= 7.5 else ("Media" if pred[0] >= 5 else "Baja"),
              delta=f"{'↑' if pred[0]>=7.5 else ('→' if pred[0]>=5 else '↓')}")
with col3:
    st.metric("Error Estimado", "±1.13 puntos", help="RMSE del modelo")

st.progress(pred[0]/10)
st.caption(f"**Interpretación:** {'Se recomiendan intervenciones' if pred[0]<5 else 'Rendimiento adecuado' if pred[0]<7.5 else 'Desempeño óptimo'}")

# --- Recomendaciones basadas en importancia ---
st.subheader("🎯 Recomendaciones Personalizadas")

def get_recommendation(feature, current_value):
    recommendations = {
        'job_satisfaction_score': [
            ("Mejorar ambiente laboral", current_value < 6),
            ("Fomentar reconocimiento", current_value < 7),
            ("Ofrecer desarrollo profesional", True)
        ],
        'sleep_hours': [
            ("Aumentar horas de sueño", current_value < 7),
            ("Mejorar higiene del sueño", current_value < 8),
            ("Reducir pantallas antes de dormir", True)
        ],
        'stress_level': [
            ("Implementar técnicas de relajación", current_value > 5),
            ("Reducir carga laboral", current_value > 6),
            ("Fomentar pausas activas", True)
        ]
    }
    return [rec[0] for rec in recommendations.get(feature, []) if rec[1]]

top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

if pred[0] < 5:
    st.warning("⚠️ **Intervenciones Críticas Requeridas**")
    for feature, imp in top_features[:3]:  # Top 3 más importantes
        current = df_input[feature].iloc[0]
        st.markdown(f"**{feature.replace('_', ' ').title()}** ({imp}% impacto)")
        for rec in get_recommendation(feature, current)[:2]:
            st.write(f"- {rec}")
            
elif pred[0] < 7.5:
    st.info("ℹ️ **Oportunidades de Mejora**")
    for feature, imp in top_features[3:]:  # Siguientes 2 importantes
        current = df_input[feature].iloc[0]
        st.markdown(f"**{feature.replace('_', ' ').title()}** ({imp}% impacto)")
        for rec in get_recommendation(feature, current)[:1]:
            st.write(f"- {rec}")
else:
    st.success("✅ **Rendimiento Óptimo**")
    st.write("Mantenga sus buenos hábitos actuales, especialmente en:")
    for feature, imp in top_features[:2]:
        st.write(f"- {feature.replace('_', ' ').title()} (actual: {df_input[feature].iloc[0]})")

# --- Simulador de Impacto ---
st.subheader("📈 Simulador de Impacto")
sim_var = st.selectbox(
    "Variable a simular", 
    options=[f[0] for f in top_features],
    format_func=lambda x: f"{x.replace('_', ' ').title()} ({feature_importance[x]}% impacto)"
)

current_val = df_input[sim_var].iloc[0]
new_val = st.slider(
    f"Nuevo valor para {sim_var.replace('_', ' ')}",
    min_value=float(0 if isinstance(current_val, (int, float)) else 0.0,
    max_value=float(current_val*2 if current_val != 0 else 10),
    value=float(current_val)
)

if new_val != current_val:
    df_sim = df_input.copy()
    df_sim[sim_var] = new_val
    sim_pred = model.predict(preprocessor.transform(df_sim))[0]
    delta = sim_pred - pred[0]
    
    st.metric(
        f"Productividad proyectada al cambiar {sim_var.replace('_', ' ')}",
        f"{sim_pred:.2f}",
        delta=f"{delta:+.2f} puntos",
        delta_color="inverse" if delta < 0 else "normal"
    )

# --- Footer ---
st.markdown("---")
st.markdown("""
**🔧 Tecnología:**  
- Modelo: Gradient Boosting (R²=0.65, RMSE=1.13)  
- Datos: 30k registros con 19 características  
""")