import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Cargar modelo y diccionario inverso
# ----------------------------
@st.cache_resource
def cargar_modelo_y_diccionario(path="best_model.pkl"):
    with open(path, "rb") as file:
        data = pickle.load(file)
        return data["model"], data["label_encoder_mapping"]

# ----------------------------
# Cargar DataFrame base
# ----------------------------
@st.cache_data
def cargar_dataframe(path="dataframe.xlsx"):
    return pd.read_excel(path, engine="openpyxl")

# ----------------------------
# Predicción
# ----------------------------
def predecir_estado(modelo, diccionario, df_base, edad, quejas, estrato):
    columnas = df_base.drop(columns="Estado Aprendiz").columns
    valores_base = df_base.drop(columns="Estado Aprendiz").mean()

    muestra = valores_base.copy()
    muestra["Edad"] = edad
    muestra["Cantidad de quejas"] = quejas
    muestra["Estrato"] = estrato

    entrada = pd.DataFrame([muestra])[columnas]
    pred_cod = modelo.predict(entrada)[0]
    return diccionario.get(pred_cod, f"Desconocido ({pred_cod})")

# ----------------------------
# Main App
# ----------------------------
def main():
    st.set_page_config(page_title="Predicción Estado del Aprendiz", page_icon="📊")
    st.title("📊 Predicción del Estado del Aprendiz")
    st.markdown("Ingrese los datos del aprendiz para realizar la predicción:")

    modelo, diccionario_inv = cargar_modelo_y_diccionario()
    df = cargar_dataframe()

    # Entradas
    edad = st.slider("Edad", 18, 100, 25)
    quejas = st.selectbox("Cantidad de quejas", list(range(11)))
    estrato = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

    if st.button("Realizar predicción"):
        try:
            resultado = predecir_estado(modelo, diccionario_inv, df, edad, quejas, estrato)
            st.success(f"✅ Estado del aprendiz predicho: **{resultado}**")
            st.markdown("### 📋 Datos ingresados:")
            st.json({
                "Edad": edad,
                "Cantidad de quejas": quejas,
                "Estrato socioeconómico": estrato
            })
        except Exception as e:
            st.error("❌ Error durante la predicción.")
            st.exception(e)

if __name__ == "__main__":
    main()
