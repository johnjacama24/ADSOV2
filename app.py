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
# Cargar el dataframe original (para estructura de columnas)
# ----------------------------
df = pd.read_excel("dataframe.xlsx", engine="openpyxl")
columnas_modelo = df.drop("Estado Aprendiz", axis=1).columns

# ----------------------------
# Configuración de la app
# ----------------------------
st.title("Predicción del Estado del Aprendiz")
st.write("Complete la información para predecir el estado del aprendiz.")

# Entradas del usuario
edad_input = st.slider("Edad", 18, 100, 25)
quejas_input = st.selectbox("Cantidad de quejas", list(range(0, 11)))
estrato_input = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

if st.button("Realizar predicción"):
    try:
        # Crear una nueva muestra con valores por defecto
        nueva_muestra = df.drop("Estado Aprendiz", axis=1).iloc[0:1].copy()

        # Identificar columnas que probablemente correspondan a los inputs
        col_edad = [col for col in columnas_modelo if "edad" in col.lower()]
        col_quejas = [col for col in columnas_modelo if "queja" in col.lower()]
        col_estrato = [col for col in columnas_modelo if "estrato" in col.lower()]

        if col_edad:
            nueva_muestra[col_edad[0]] = edad_input
        if col_quejas:
            nueva_muestra[col_quejas[0]] = quejas_input
        if col_estrato:
            nueva_muestra[col_estrato[0]] = estrato_input

        # Predecir
        pred = modelo.predict(nueva_muestra)[0]
        resultado = diccionario_inverso.get(pred, f"Desconocido ({pred})")

        # Mostrar resultado
        st.subheader("Resultado de la predicción:")
        st.success(f"📊 Estado del aprendiz predicho: **{resultado}**")

        st.subheader("Valores utilizados:")
        st.write({
            col_edad[0] if col_edad else "Edad": edad_input,
            col_quejas[0] if col_quejas else "Cantidad de quejas": quejas_input,
            col_estrato[0] if col_estrato else "Estrato": estrato_input
        })

    except Exception as e:
        st.error("❌ Error al hacer la predicción:")
        st.code(str(e))
