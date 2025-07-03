# Heart Disease Prediction

Este proyecto utiliza Machine learning para predecir la probabilidad de enfermedad cardíaca a partir de datos médicos del usuario. Se presenta una aplicación web interactiva desarrollada en Streamlit, que facilita tanto la entrada de datos como la visualización de resultados. El modelo ha sido entrenado y normalizado previamente, y permite su uso tanto en local como en la nube.

## Contexto del Proyecto

La predicción temprana de enfermedades cardíacas puede salvar vidas. Este sistema permite a médicos o usuarios ingresar parámetros clave relacionados con la salud cardiovascular y recibir una predicción inmediata sobre el riesgo de enfermedad cardíaca, junto con el nivel de confianza del modelo.

## Estructura del Repositorio

```
Heart-Disease-Prediction/
├── app.py                  # Aplicación principal de Streamlit
├── models/
│   ├── model.pkl           # Modelo de machine learning serializado
│   └── mean_std_values.pkl # Valores de media y desviación estándar para normalización
├── notebooks/              # Notebooks para análisis, entrenamiento y experimentación
├── README.md               # Este archivo
```

## Librerías usadas

- `streamlit`: Para la creación de la interfaz web interactiva.
- `pandas`: Manejo y transformación de datos.
- `pickle`: Carga del modelo y parámetros serializados.
- `os`: Manejo de rutas y archivos.

## Cómo ejecutar en local

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/SamilUPAO/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. **Asegúrate de que los archivos del modelo estén en la carpeta `models/`:**
   - `models/model.pkl`
   - `models/mean_std_values.pkl`

3. **Ejecuta la aplicación:**
   ```bash
   streamlit run app.py
   ```

## Enlace de despliegue

Puedes probar la aplicación desplegada aquí:  
[https://heart-disease-prediction-iapt-upao.streamlit.app/](https://heart-disease-prediction-iapt-upao.streamlit.app/)

## Integrantes / Colaboradores

1. Aroni Muñoz, Francisco  
2. Castillo Pezo, Mateo  
3. Cruz León, Gustavo  
4. Grados Araujo, Samil  
5. Liu Dai, Yan  
6. Marquina Laguna, Abraham  
