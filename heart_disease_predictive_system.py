# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import pickle

loaded_model = pickle.load(open('heart_disease_prediction_model.sav', 'rb'))
sc = pickle.load(open('scaler.sav', 'rb'))

column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

scaled_features = [3, 4, 7, 9]

input_data = np.array([[52, 1, 3, 152, 298, 1, 1, 178, 0, 1.2, 1, 0, 3]])

input_df = pd.DataFrame(input_data, columns=column_names)

input_df.iloc[:, scaled_features] = sc.transform(input_df.iloc[:, scaled_features])

input_data_scaled = input_df.values

prediction = loaded_model.predict(input_data_scaled)
print(prediction)

if prediction == 1:
    print('The person has heart disease')
else:
    print('The person does not have heart disease')
