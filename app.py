import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Cargar el modelo y datos desde el archivo .pkl
# ----------------------------
@st.cache_resource
def cargar_modelo_y_datos():
    with open("best_model.pkl", "rb") as file:
        data = pickle.load(file)
        modelo = data["model"]
        diccionario_inverso = data["label_encoder_mapping"]
        dataframe_entrenamiento = data["dataframe"]  # Asegúrate que este key existe en tu .pkl
        return modelo, diccionario_inverso, dataframe_entrenamiento

modelo, diccionario_inverso, df = cargar_modelo_y_datos()

# ----------------------------
# Configuración de la app
# ----------------------------
st.title("🔍 Predicción del Estado del Aprendiz")
st.write("Ingrese los datos necesarios para realizar una predicción basada en el modelo entrenado.")

# ----------------------------
# Entradas del usuario
# ----------------------------
edad = st.slider("Edad", 18, 100, 25)
cantidad_quejas = st.selectbox("Cantidad de quejas", list(range(0, 11)))
estrato = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

# ----------------------------
# Botón de predicción
# ----------------------------
if st.button("Realizar predicción"):
    try:
        columnas_modelo = df.drop("Estado Aprendiz", axis=1).columns
        valores_base = df.drop("Estado Aprendiz", axis=1).mean()

        muestra = valores_base.copy()
        muestra["Edad"] = edad
        muestra["Cantidad de quejas"] = cantidad_quejas
        muestra["Estrato"] = estrato

        entrada = pd.DataFrame([muestra])[columnas_modelo]

        prediccion_codificada = modelo.predict(entrada)[0]
        prediccion = diccionario_inverso.get(prediccion_codificada, f"Desconocido ({prediccion_codificada})")

        st.subheader("📈 Resultado de la predicción:")
        st.success(f"Estado del aprendiz predicho: **{prediccion}**")

        st.subheader("📌 Datos ingresados:")
        st.write({
            "Edad": edad,
            "Cantidad de quejas": cantidad_quejas,
            "Estrato socioeconómico": estrato
        })

    except Exception as error:
        st.error("❌ Error al realizar la predicción:")
        st.code(str(error))

