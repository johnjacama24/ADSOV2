import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Predicción Estado del Aprendiz", page_icon="📊")

# ----------------------------
# Cargar modelo y diccionario
# ----------------------------
@st.cache_resource
def cargar_modelo_y_diccionario(path="best_model.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    modelo = data.get("model")
    
    posibles_claves = [
        "label_encoder_mapping", "diccionario_inverso", 
        "diccionario_inverso_estado_aprendiz", "inverse_mapping"
    ]
    diccionario = next((data[k] for k in posibles_claves if k in data), None)
    
    if modelo is None:
        raise ValueError("No se encontró el modelo en el archivo.")
    
    return modelo, diccionario

# ----------------------------
# Cargar DataFrame
# ----------------------------
@st.cache_data
def cargar_dataframe(path="dataframe.xlsx"):
    return pd.read_excel(path, engine="openpyxl")

# ----------------------------
# Predicción del estado
# ----------------------------
def predecir_estado(modelo, diccionario, df_base, edad, quejas, estrato):
    # Obtener las columnas que el modelo espera
    columnas_modelo = list(modelo.feature_names_in_)
    
    # Calcular valores promedio de esas columnas desde el DataFrame base
    valores_prom = df_base[columnas_modelo].mean(numeric_only=True)

    # Sobrescribir solo si esas columnas existen
    if "Edad" in columnas_modelo:
        valores_prom["Edad"] = edad
    if "Cantidad de quejas" in columnas_modelo:
        valores_prom["Cantidad de quejas"] = quejas
    if "Estrato" in columnas_modelo:
        valores_prom["Estrato"] = estrato

    # Construir la muestra
    muestra = pd.DataFrame([valores_prom])[columnas_modelo]

    # Predicción
    pred = modelo.predict(muestra)[0]
    if diccionario:
        return diccionario.get(pred, f"Desconocido ({pred})")
    return f"(Sin mapeo) Clase codificada: {pred}"

# ----------------------------
# Interfaz principal
# ----------------------------
def main():
    st.title("🔍 Predicción del Estado del Aprendiz")
    st.markdown("Ingrese los datos del aprendiz para predecir su estado formativo.")

    try:
        modelo, diccionario = cargar_modelo_y_diccionario()
        df = cargar_dataframe()
    except Exception as e:
        st.error(f"Error al cargar modelo o datos: {e}")
        return

    edad = st.slider("Edad", 18, 100, 25)
    quejas = st.selectbox("Cantidad de quejas", list(range(11)))
    estrato = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

    if st.button("Realizar predicción"):
        try:
            resultado = predecir_estado(modelo, diccionario, df, edad, quejas, estrato)
            st.success(f"📊 Estado del aprendiz predicho: **{resultado}**")
            st.markdown("### 📋 Datos utilizados")
            st.json({
                "Edad": edad,
                "Cantidad de quejas": quejas,
                "Estrato": estrato
            })
        except Exception as e:
            st.error(f"❌ Error durante la predicción.")
            st.exception(e)

if __name__ == "__main__":
    main()
