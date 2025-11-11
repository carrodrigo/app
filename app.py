# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ------------------------------
# Diccionario de modelos y OHE por programa
# ------------------------------
modelos_paths = {
    "Administración": {"modelo": "admin_pipeline (1).pkl", "ohe": "admin_orde.pkl"},
    "Contaduría": {"modelo": "Contaduria_pipeline_final.pkl", "ohe": "Contaduria_one_hot_encoder.pkl"},
    "Derecho": {"modelo": "Derecho_pipeline_final.pkl", "ohe": "Derecho_one_hot_encoder.pkl"},
    "Comercio": {"modelo": "comercio_pipeline (1).pkl", "ohe": "comercio_orde.pkl"},
    "Comunicación Social": {"modelo": "comunicacion_pipeline.pkl", "ohe": "comunicacion_ohe.pkl"},
    "Psicología": {"modelo": "psico_pipeline.pkl", "ohe": "psico_ohe.pkl"}
}

# ------------------------------
# Funciones de agrupación
# ------------------------------
def agrupar_fuente(fuente):
    grupos = {
        'Medios_Digitales_Online': ['Página Web', 'FACEBOOK', 'Link', 'Correo Directo',
                                    'Pantalla Publicitaria Led', 'Banner', 'Instagram'],
        'Redes_Personales_Boca_Boca': ['Amigos', 'Familiares', 'Estudiante', 'Docente',
                                       'Egresado', 'Funcionario', 'Referido Universidad Cooperat', 'Embajador Corpaeda'],
        'Eventos_Ferias': ['Feria Universitaria', 'Visita a Colegi', 'Guías Orientación Profesional'],
        'Medios_Tradicionales_Masivos': ['Prensa', 'Televisión', 'Radio', 'Valla', 'Pasacalle',
                                         'Paradero Transporte', 'Transporte Masivo', 'Directorio Telefónico']
    }
    for grupo, valores in grupos.items():
        if fuente in valores:
            return grupo
    return "Otro"


def agrupar_pago(pago):
    grupos = {
        'Credito_Financ_Externa': ['Crédito ICETEX', 'Financ por Banco o Cooperativa', 'Crédito Coop. Comuna'],
        'Pago_Directo_Propio': ['Contado / Efectivo', 'Tarjeta de Crédito', 'Fondo de Cesantias'],
        'Apoyo_Beneficio': ['Beca', 'Subsidio Empresarial'],
        "Otro": ["Otro"]
    }
    for grupo, valores in grupos.items():
        if pago in valores:
            return grupo
    return "Otro"


# ------------------------------
# Configuración de la app
# ------------------------------
st.set_page_config(page_title="Prototipo de Predicción de Matrícula", page_icon="🎓", layout="centered")

st.markdown("<h1 style='text-align:center; color:#d62828;'>Prototipo de Predicción de matrícula</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#003049;'>Aspirantes a Programas Académicos</h3>", unsafe_allow_html=True)
st.write("Selecciona el programa y completa los datos del aspirante para obtener la predicción.")

# ------------------------------
# Selección del programa
# ------------------------------
programa = st.selectbox("Programa académico", list(modelos_paths.keys()))

modelo = joblib.load(modelos_paths[programa]["modelo"])
ohe_cargado = joblib.load(modelos_paths[programa]["ohe"])

# ------------------------------
# Entrada de datos
# ------------------------------
col1, col2 = st.columns([1, 2])

# Columna izquierda: sliders
with col1:
    matematicas = st.slider("Puntaje Matemáticas", 0, 100, 60)
    ciencias = st.slider("Puntaje Ciencias", 0, 100, 50)
    ingles = st.slider("Puntaje Inglés", 0, 100, 60)
    lectura = st.slider("Puntaje Lectura Crítica", 0, 100, 55)
    sociales = st.slider("Puntaje Sociales", 0, 100, 70)

# Columna derecha: dos subcolumnas
with col2:
    col2a, col2b = st.columns(2)

    with col2a:
        estrato = st.selectbox("Estrato", [1, 2, 3, 4, 5, 6])
        trabaja = st.selectbox("Trabaja Actualmente", ["Si", "No"])
        edad = st.number_input("Edad de inscripción", min_value=15, max_value=70, value=18)
        distancia = st.number_input("Distancia a la universidad (km)", min_value=0, max_value=200, value=10)

    with col2b:
        posible_pago_raw = st.selectbox("Posible forma de pago", [
            "Contado / Efectivo", "Tarjeta de Crédito", "Fondo de Cesantias",
            "Crédito ICETEX", "Financ por Banco o Cooperativa",
            "Crédito Coop. Comuna", "Beca", "Subsidio Empresarial", "Otro"
        ])
        fuente_raw = st.selectbox("Fuente de referencia", [
            "Página Web", "FACEBOOK", "Correo Directo", "Banner", "Instagram",
            "Amigos", "Familiares", "Docente", "Egresado", "Feria Universitaria",
            "Televisión", "Radio", "Otro"
        ])
        anio = st.number_input("Año de inscripción", min_value=2020, max_value=2030, value=2025)
        semestre = st.selectbox("Semestre", ["01", "02"])

# ------------------------------
# Preprocesamiento
# ------------------------------
df_nuevo = pd.DataFrame({
    'Edad inscripcion': [edad],
    'Ciencias': [ciencias],
    'Inglés': [ingles],
    'Lectura Crítica': [lectura],
    'Matematicas': [matematicas],
    'Sociales': [sociales],
    'Distancia a Universidad (km)': [distancia],
    'Año': [anio],
    'Trabaja Actualmente': [trabaja],
    'Estrato': [estrato],
    'Fuente Referencia': [agrupar_fuente(fuente_raw)],
    'Posible Forma de Pago': [agrupar_pago(posible_pago_raw)],
    'Semestre': [semestre]
})

categorical_cols = ['Trabaja Actualmente', 'Estrato', 'Fuente Referencia', 'Posible Forma de Pago', 'Semestre']
for col in categorical_cols:
    df_nuevo[col] = df_nuevo[col].astype(str)

datos_encoded = ohe_cargado.transform(df_nuevo[categorical_cols])
encoded_df = pd.DataFrame(datos_encoded, columns=ohe_cargado.get_feature_names_out())

df_final = pd.concat([
    df_nuevo.drop(columns=categorical_cols).reset_index(drop=True),
    encoded_df.reset_index(drop=True)
], axis=1)

if hasattr(modelo, "feature_names_in_"):
    df_final = df_final.reindex(columns=modelo.feature_names_in_, fill_value=0)

# ------------------------------
# Predicción
# ------------------------------
if st.button("Predecir"):
    try:
        pred = modelo.predict(df_final)[0]
        score = modelo.predict_proba(df_final)[0][1]

        st.markdown("---")
        st.markdown("### Resultado de la predicción")

        colr1, colr2 = st.columns([1, 2])

        with colr1:
            st.markdown(f"**Clase predicha:** {'✅ Matrícula' if pred == 1 else '❌ Admisión'}")

        with colr2:
            st.write(f"**Puntuación:** {score:.2f}")

    except Exception as e:
        st.error(f"Error en la predicción: {e}")
