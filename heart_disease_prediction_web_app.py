# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:00:24 2024

@author: jivan
"""

import numpy as np
import pickle
import streamlit as st
from PIL import Image

loaded_model = pickle.load(open('C:/Users/jivan/OneDrive/Desktop/ml model deployment/heart_disease_prediction_model.sav', 'rb'))
sc = pickle.load(open('C:/Users/jivan/OneDrive/Desktop/ml model deployment/scaler.sav', 'rb'))

def heart_disease_prediction(in_data):
    scaled_features = [3, 4, 7, 9]
    input_data = np.array([in_data])

    input_data[:, scaled_features] = sc.transform(input_data[:, scaled_features])

    prediction = loaded_model.predict(input_data)

    if prediction == 1:
        return 'The person has heart disease'
    else:
        return 'The person does not have heart disease'

def main():
    st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout='centered', initial_sidebar_state='expanded')
    
    st.title("Heart Disease Prediction")
    st.markdown("## Predict your heart health with our machine learning model")
    st.image(Image.open('C:/Users/jivan/OneDrive/Desktop/ml model deployment/heart_image.jpg'), use_column_width=True)

    st.sidebar.header("User Input Parameters")
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=52)
    sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
    sex = 1 if sex == "Male" else 0
    cp = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=130)
    chol = st.sidebar.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=230)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    fbs = 1 if fbs == "Yes" else 0
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", ["Normal", "Having ST-T Wave Abnormality", "Showing Probable or Definite Left Ventricular Hypertrophy"])
    restecg = ["Normal", "Having ST-T Wave Abnormality", "Showing Probable or Definite Left Ventricular Hypertrophy"].index(restecg)
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=170)
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exang = 1 if exang == "Yes" else 0
    oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=2.3, format="%.1f")
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
    ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect", "Null"])
    thal = ["Normal", "Fixed Defect", "Reversible Defect", "Null"].index(thal)

    inp_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    
    if st.button("Predict"):
        diagnosis = heart_disease_prediction(inp_data)
        st.markdown(f"### Result: {diagnosis}")

if __name__ == '__main__':
    main()
