import streamlit as st 
import pandas as pd
import pickle

# ----------------------------
# Cargar el modelo y diccionario del .pkl
# ----------------------------
@st.cache_resource
def cargar_modelo_y_diccionario():
    with open("best_model.pkl", "rb") as file:
        data = pickle.load(file)
        return data["model"], data["label_encoder_mapping"]

modelo, diccionario_inverso = cargar_modelo_y_diccionario()

# ----------------------------
# Cargar el dataframe original
# ----------------------------
df = pd.read_excel("dataframe.xlsx", engine="openpyxl")

# ----------------------------
# Configuración de la app
# ----------------------------
st.title("Predicción del Estado del Aprendiz")
st.write("Complete la información para predecir el estado del aprendiz.")

# Entradas del usuario
edad = st.slider("Edad", 18, 100, 25)
cantidad_quejas = st.selectbox("Cantidad de quejas", list(range(0, 11)))
estrato = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

# Botón para ejecutar la predicción
if st.button("Realizar predicción"):
    try:
        # Quitar la columna objetivo
        df_features = df.drop("Estado Aprendiz", axis=1)

        # Crear una nueva muestra con los promedios
        nueva_muestra = df_features.mean()

        # Buscar columnas reales asociadas a cada variable
        columnas_edad = [col for col in df_features.columns if "edad" in col.lower()]
        columnas_quejas = [col for col in df_features.columns if "quejas" in col.lower()]
        columnas_estrato = [col for col in df_features.columns if "estrato" in col.lower()]

        # Reemplazar valores ingresados
        if columnas_edad:
            nueva_muestra[columnas_edad[0]] = edad
        if columnas_quejas:
            nueva_muestra[columnas_quejas[0]] = cantidad_quejas
        if columnas_estrato:
            nueva_muestra[columnas_estrato[0]] = estrato

        # Convertir a DataFrame y reordenar columnas
        entrada_modelo = pd.DataFrame([nueva_muestra])[df_features.columns]

        # Realizar la predicción
        prediccion_codificada = modelo.predict(entrada_modelo)[0]
        prediccion_original = diccionario_inverso.get(prediccion_codificada, f"Desconocido ({prediccion_codificada})")

        # Mostrar resultado
        st.subheader("Resultado de la predicción:")
        st.success(f"📊 Estado del aprendiz predicho: **{prediccion_original}**")

        # Mostrar entradas originales
        st.subheader("Valores utilizados para la predicción:")
        st.write({
            columnas_edad[0] if columnas_edad else "Edad": edad,
            columnas_quejas[0] if columnas_quejas else "Cantidad de quejas": cantidad_quejas,
            columnas_estrato[0] if columnas_estrato else "Estrato": estrato
        })

    except Exception as e:
        st.error("❌ Error al hacer la predicción:")
        st.code(str(e))
