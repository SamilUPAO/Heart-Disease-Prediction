import streamlit as st
import pandas as pd
import pickle

model_filename = './models/model.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

def main():
    st.title('Predicción de Enfermedad Cardiaca')
    age = st.slider('Edad', 18, 100, 50)
    sex_options = ['Masculino', 'Femenino']
    sex = st.selectbox('Sexo', sex_options)
    sex_num = 1 if sex == 'Masculino' else 0
    cp_options = ['Angina Típica', 'Angina Atypical', 'Dolor No Anginal', 'Asintomático']
    cp = st.selectbox('Tipo de Dolor en el Pecho', cp_options)
    cp_num = cp_options.index(cp)
    trestbps = st.slider('Presión Arterial en Reposo', 90, 200, 120)
    chol = st.slider('Colesterol', 100, 600, 250)
    fbs_options = ['Falso', 'Verdadero']
    fbs = st.selectbox('Azúcar en sangre en ayunas > 120 mg/dl', fbs_options)
    fbs_num = fbs_options.index(fbs)
    restecg_options = ['Normal', 'Anormalidad ST-T', 'Hipertrofia Ventricular Izquierda']
    restecg = st.selectbox('Resultados Electrocardiográficos en Reposo', restecg_options)
    restecg_num = restecg_options.index(restecg)
    thalach = st.slider('Frecuencia Cardíaca Máxima Alcanzada', 70, 220, 150)
    exang_options = ['No', 'Sí']
    exang = st.selectbox('Angina Inducida por Ejercicio', exang_options)
    exang_num = exang_options.index(exang)
    oldpeak = st.slider('Depresión ST Inducida por Ejercicio en Relación al Reposo', 0.0, 6.2, 1.0)
    slope_options = ['Pendiente ascendente', 'Plana', 'Pendiente descendente']
    slope = st.selectbox('Pendiente del Segmento ST en Ejercicio Pico', slope_options)
    slope_num = slope_options.index(slope)
    ca = st.slider('Número de Vasos Principales Coloreados por Fluoroscopia', 0, 4, 1)
    thal_options = ['Normal', 'Defecto Fijo', 'Defecto Reversible']
    thal = st.selectbox('Talasemia', thal_options)
    thal_num = thal_options.index(thal)

    with open('models/mean_std_values.pkl', 'rb') as f:
        mean_std_values = pickle.load(f)


    if st.button('Predict'):
        user_input = pd.DataFrame(data={
            'age': [age],
            'sex': [sex_num],  
            'cp': [cp_num],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs_num],
            'restecg': [restecg_num],
            'thalach': [thalach],
            'exang': [exang_num],
            'oldpeak': [oldpeak],
            'slope': [slope_num],
            'ca': [ca],
            'thal': [thal_num]
        })
        # Apply saved transformation to new data
        user_input = (user_input - mean_std_values['mean']) / mean_std_values['std']
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        if prediction[0] == 1:
            bg_color = 'red'
            prediction_result = 'Positivo'
        else:
            bg_color = 'green'
            prediction_result = 'Negativo'

        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

        st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Predicción: {prediction_result}<br>Confianza: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()