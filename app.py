# app.py
import streamlit as st 
import pandas as pd
import pickle
import re

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
# Normalizar nombres de columnas
# ----------------------------
def normalizar_columnas(cols):
    return cols.str.strip()\
               .str.replace('–', '-', regex=False)\
               .str.replace('—', '-', regex=False)\
               .str.replace(r'[.,;:¡!¿?()]', '', regex=True)\
               .str.replace(r'\\s+', ' ', regex=True)\
               .str.replace(r' +', ' ', regex=True)\
               .str.upper()

# ----------------------------
# Cargar y procesar el dataframe
# ----------------------------
df = pd.read_excel("dataframe.xlsx", engine="openpyxl")
df.columns = normalizar_columnas(df.columns)

# Identificar la columna de estado
columna_estado = [col for col in df.columns if "ESTADO APRENDIZ" in col]
if not columna_estado:
    st.error("❌ No se encontró la columna 'Estado Aprendiz' tras la normalización.")
    st.stop()
col_estado = columna_estado[0]

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
        columnas_modelo = normalizar_columnas(df.drop(col_estado, axis=1).columns)

        # Crear muestra con valores promedio
        valores_default = df.drop(col_estado, axis=1).mean()
        nueva_muestra = valores_default.copy()

        # Reemplazar los valores ingresados
        nueva_muestra["EDAD"] = edad
        nueva_muestra["CANTIDAD DE QUEJAS"] = cantidad_quejas
        nueva_muestra["ESTRATO"] = estrato

        # Normalizar columnas de la muestra
        nueva_muestra.index = normalizar_columnas(nueva_muestra.index)

        # Convertir en DataFrame con columnas en el orden original
        entrada_modelo = pd.DataFrame([nueva_muestra])[columnas_modelo]

        # Verificación de columnas faltantes
        columnas_actuales = entrada_modelo.columns
        faltantes = set(columnas_modelo) - set(columnas_actuales)
        if faltantes:
            st.error(f"Columnas faltantes en la predicción: {faltantes}")
        else:
            # Realizar la predicción
            prediccion_codificada = modelo.predict(entrada_modelo)[0]
            prediccion_original = diccionario_inverso.get(prediccion_codificada, f"Desconocido ({prediccion_codificada})")

            # Mostrar resultado
            st.subheader("Resultado de la predicción:")
            st.success(f"📊 Estado del aprendiz predicho: **{prediccion_original}**")

            # Mostrar entradas originales
            st.subheader("Valores utilizados para la predicción:")
            st.write({
                "Edad": edad,
                "Cantidad de quejas": cantidad_quejas,
                "Estrato socioeconómico": estrato
            })

    except Exception as e:
        st.error("❌ Error al hacer la predicción:")
        st.code(str(e))
