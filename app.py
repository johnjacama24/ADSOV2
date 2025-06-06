# app.py
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
# Cargar el DataFrame sin modificar nombres
# ----------------------------
df = pd.read_excel("dataframe.xlsx", engine="openpyxl")

# Detectar automáticamente la columna de estado
columna_estado = [col for col in df.columns if "Estado Aprendiz" in col]
if not columna_estado:
    st.error("❌ No se encontró una columna que contenga 'Estado Aprendiz'.")
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
        columnas_modelo = df.drop(columns=[col_estado]).columns

        # Crear muestra con valores promedio
        valores_default = df.drop(columns=[col_estado]).mean()
        nueva_muestra = valores_default.copy()

        # Reemplazar los valores ingresados
        nueva_muestra["Edad"] = edad
        nueva_muestra["Cantidad de quejas"] = cantidad_quejas
        nueva_muestra["Estrato"] = estrato

        # Convertir en DataFrame con columnas en el orden original
        entrada_modelo = pd.DataFrame([nueva_muestra])[list(columnas_modelo)]

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
