import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Cargar el modelo, el diccionario y el dataframe codificado
# ----------------------------
@st.cache_resource
def cargar_modelo_datos():
    with open("best_model.pkl", "rb") as file:
        data = pickle.load(file)
        modelo = data["model"]
        diccionario_inverso = data["label_encoder_mapping"]
        df = data["dataframe_codificado"]  # <- nombre correcto según tu indicación
        return modelo, diccionario_inverso, df

modelo, diccionario_inverso, df_codificado = cargar_modelo_datos()

# ----------------------------
# Configuración de la App
# ----------------------------
st.title("🔍 Predicción del Estado del Aprendiz")
st.write("Ingrese los datos solicitados para realizar una predicción basada en el modelo entrenado.")

# Entradas del usuario
edad = st.slider("Edad", 18, 100, 25)
cantidad_quejas = st.selectbox("Cantidad de quejas", list(range(0, 11)))
estrato = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

# ----------------------------
# Botón para predecir
# ----------------------------
if st.button("Realizar predicción"):
    try:
        # Obtener las columnas esperadas por el modelo (sin la columna objetivo)
        columnas_modelo = df_codificado.drop("Estado Aprendiz", axis=1).columns

        # Calcular promedio por columna
        promedio_columnas = df_codificado.drop("Estado Aprendiz", axis=1).mean()

        # Crear muestra base con promedios
        muestra = promedio_columnas.copy()

        # Reemplazar con valores del usuario
        if "Edad" in muestra:
            muestra["Edad"] = edad
        if "Cantidad de quejas" in muestra:
            muestra["Cantidad de quejas"] = cantidad_quejas
        if "Estrato" in muestra:
            muestra["Estrato"] = estrato

        # Armar DataFrame de entrada con las columnas originales
        entrada = pd.DataFrame([muestra])[columnas_modelo]

        # Realizar predicción
        pred_codificada = modelo.predict(entrada)[0]
        pred_original = diccionario_inverso.get(pred_codificada, f"Desconocido ({pred_codificada})")

        # Mostrar resultados
        st.subheader("📈 Resultado de la predicción:")
        st.success(f"Estado del aprendiz predicho: **{pred_original}**")

        st.subheader("📌 Datos ingresados:")
        st.write({
            "Edad": edad,
            "Cantidad de quejas": cantidad_quejas,
            "Estrato socioeconómico": estrato
        })

    except Exception as e:
        st.error("❌ Error al hacer la predicción:")
        st.code(str(e))

