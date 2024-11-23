import streamlit as st
import pandas as pd
import joblib
import os

# Configuración de la página
st.set_page_config(layout="wide")

# CSS para cambiar el color de los encabezados de los campos
st.markdown("""
<style>
/* Cambiar el color de las etiquetas de los campos de entrada */
label {
    color: white !important;  /* Cambiar a color blanco */
    font-size: 16px;          /* Ajustar tamaño de fuente */
    font-weight: bold;        /* Negrita */
}
</style>
""", unsafe_allow_html=True)

# Función para agregar la imagen de fondo
def agregar_imagen_fondo():
    import base64
    from pathlib import Path

    image_path = Path("imagenes/fondo.jpeg")
    if not image_path.exists():
        st.error("No se encontró la imagen de fondo en la carpeta 'imagenes/'. Verifica la ruta y el nombre del archivo.")
        return

    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
            fondo_css = f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{encoded_image});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """
            st.markdown(fondo_css, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error al cargar la imagen de fondo: {str(e)}")

# Llamar a la función para añadir la imagen de fondo
agregar_imagen_fondo()

# Cargar el modelo y las columnas usadas durante el entrenamiento
try:
    modelo = joblib.load('modelo_random_forest.pkl')  # Modelo entrenado
    columnas_modelo = joblib.load('columnas_modelo.pkl')  # Columnas esperadas por el modelo
except FileNotFoundError:
    st.error("No se encontraron los archivos necesarios (modelo o columnas). Verifica su existencia.")
    st.stop()

# Ruta del archivo CSV
base_path = os.path.dirname(__file__)
tabla_path = os.path.join(base_path, 'datos', 'Tabla_Final.csv')

# Cargar el archivo Tabla_Final.csv para obtener provincias y diagnósticos únicos
try:
    tabla_final = pd.read_csv(tabla_path)
    provincias = sorted(tabla_final['Provincia'].unique())  # Lista de provincias únicas
    diagnosticos = sorted(tabla_final['Diagnóstico'].unique())  # Lista de diagnósticos únicos
except FileNotFoundError:
    st.error("No se encontró el archivo 'Tabla_Final.csv'. Verifica que esté en la carpeta 'datos'.")
    st.stop()

# Crear columnas con proporciones para maximizar el ancho de los cuadros
col1, col_space = st.columns([1, 2])

# Cuadro de predicción en la columna izquierda
with col1:
    st.markdown("<h2 style='text-align: left; color: white;'>Cuadro de Predicción</h2>", unsafe_allow_html=True)

    # Selectores de entradas categóricas
    provincia = st.selectbox("Selecciona la Provincia", provincias)
    diagnostico = st.selectbox("Selecciona el Diagnóstico", diagnosticos)
    sexo = st.selectbox("Selecciona el Sexo", ["Hombres", "Mujeres"])

    # Entradas numéricas
    habitantes = st.number_input("Número de Habitantes", min_value=0, step=100)
    metales_pesados = st.number_input("Metales Pesados (As + Cd + Ni + Pb)", min_value=0.0, step=0.0001)
    indice_contaminacion = st.number_input("Índice de Contaminación", min_value=0.0, step=0.1)

    # Crear DataFrame con las selecciones del usuario y convertirlas a variables dummies
    input_data = pd.DataFrame({
        'Habitantes': [habitantes],
        'Metales Pesados': [metales_pesados],
        'Indice_Contaminación': [indice_contaminacion],
        f'Diagnóstico_{diagnostico}': [1],
        f'Provincia_{provincia}': [1],
        f'Sexo_{sexo}': [1]
    })

    # Asegurar que todas las columnas usadas en el modelo están presentes
    for col in columnas_modelo:
        if col not in input_data.columns:
            input_data[col] = 0  # Rellenar las columnas faltantes con 0
    input_data = input_data[columnas_modelo]  # Reordenar las columnas

    # Botón para predecir
    st.markdown("<div style='display: flex; justify-content: center; align-items: center; margin: 10px auto;'>", unsafe_allow_html=True)
    if st.button("Predecir"):
        try:
            # Validar que los campos numéricos sean correctos
            if habitantes == 0 or metales_pesados <= 0.0 or indice_contaminacion <= 0.0:
                mensaje_error = """
                <div style='background-color: #ffcccc; color: #b30000; padding: 10px; border-radius: 10px; text-align: center; font-weight: bold;'>
                     COMPLETAR TODOS LOS CAMPOS
                </div>
                """
                st.markdown(mensaje_error, unsafe_allow_html=True)
            else:
                # Realizar la predicción
                prediccion = modelo.predict(input_data)

                # HTML personalizado para mostrar el resultado
                resultado_html = f"""<div style='display: flex; justify-content: center; align-items: center; background-color: #f0f0f0; width: 50%; height: 50px; padding: 10px; \
                border-radius: 10px; border: 2px solid #1d741b; margin:20px auto;'> <h2 style='color: #1d741b; text-align: center; font-weight: bold; text-transform: uppercase; font-size: 16px;'>
                PRONÓSTICO DE HOSPITALIZACIONES: {int(prediccion[0])}
                </h2></div>"""

                # Mostrar el resultado
                st.markdown(resultado_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {str(e)}")

# Columna central mínima para separación
with col_space:
    st.markdown("<div style='height: 100%;'></div>", unsafe_allow_html=True)