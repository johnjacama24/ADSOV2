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
edad_input = st.slider("Edad", 18, 100, 25)
quejas_input = st.selectbox("Cantidad de quejas", list(range(0, 11)))
estrato_input = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

# Botón para ejecutar la predicción
if st.button("Realizar predicción"):
    try:
        # Obtener solo las columnas de entrada esperadas
        columnas_entrada = df.drop("Estado Aprendiz", axis=1).columns

        # Crear muestra tomando una fila válida
        nueva_muestra = df.iloc[0].drop("Estado Aprendiz").copy()

        # Reemplazar valores en columnas específicas
        for col in columnas_entrada:
            if "edad" in col.lower():
                nueva_muestra[col] = edad_input
            elif "queja" in col.lower():
                nueva_muestra[col] = quejas_input
            elif "estrato" in col.lower():
                nueva_muestra[col] = estrato_input

        # Crear DataFrame de una sola fila
        entrada_modelo = pd.DataFrame([nueva_muestra])[columnas_entrada]

        # Realizar la predicción
        pred = modelo.predict(entrada_modelo)[0]
        resultado = diccionario_inverso.get(pred, f"Desconocido ({pred})")

        # Mostrar resultado
        st.subheader("Resultado de la predicción:")
        st.success(f"📊 Estado del aprendiz predicho: **{resultado}**")

        st.subheader("Valores utilizados:")
        st.write({
            "Edad": edad_input,
            "Cantidad de quejas": quejas_input,
            "Estrato": estrato_input
        })

    except Exception as e:
        st.error("❌ Error al hacer la predicción:")
        st.code(str(e))

st.write("Columnas esperadas por el modelo:")
st.write(list(df.drop("Estado Aprendiz", axis=1).columns))
