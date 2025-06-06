import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Cargar el modelo y diccionario
# ----------------------------
@st.cache_resource
def cargar_modelo_y_diccionario(path_modelo="best_model.pkl"):
    with open(path_modelo, "rb") as file:
        data = pickle.load(file)
        return data["model"], data["label_encoder_mapping"]

# ----------------------------
# Cargar el DataFrame base
# ----------------------------
@st.cache_data
def cargar_dataframe(path_df="dataframe.xlsx"):
    return pd.read_excel(path_df, engine="openpyxl")

modelo, diccionario_inverso = cargar_modelo_y_diccionario()
df = cargar_dataframe()

# ----------------------------
# Interfaz de usuario
# ----------------------------
st.title("🔍 Predicción del Estado del Aprendiz")
st.markdown("Complete los siguientes datos para predecir el **estado del aprendiz**:")

# Entradas del usuario
edad = st.slider("Edad", 18, 100, 25)
cantidad_quejas = st.selectbox("Cantidad de quejas", list(range(11)))
estrato = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

# ----------------------------
# Lógica de predicción
# ----------------------------
if st.button("Realizar predicción"):
    try:
        columnas = df.drop(columns="Estado Aprendiz").columns
        valores_promedio = df.drop(columns="Estado Aprendiz").mean()

        # Preparar entrada con valores promedio y sobrescribir los ingresados
        muestra = valores_promedio.copy()
        muestra["Edad"] = edad
        muestra["Cantidad de quejas"] = cantidad_quejas
        muestra["Estrato"] = estrato

        entrada = pd.DataFrame([muestra])[columnas]
        prediccion_codificada = modelo.predict(entrada)[0]
        prediccion_original = diccionario_inverso.get(prediccion_codificada, f"Desconocido ({prediccion_codificada})")

        # Resultados
        st.success(f"📊 Estado del aprendiz predicho: **{prediccion_original}**")
        st.markdown("**Valores utilizados para la predicción:**")
        st.json({
            "Edad": edad,
            "Cantidad de quejas": cantidad_quejas,
            "Estrato socioeconómico": estrato
        })

    except Exception as e:
        st.error("❌ Ocurrió un error durante la predicción:")
        st.exception(e)
