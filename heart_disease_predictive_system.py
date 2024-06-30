# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/jivan/OneDrive/Desktop/ml model deployment/heart_disease_prediction_model.sav', 'rb'))
sc = pickle.load(open('C:/Users/jivan/OneDrive/Desktop/ml model deployment/scaler.sav', 'rb'))

scaled_features = [3, 4, 7, 9]

input_data = np.array([[52, 1, 3, 152, 298, 1, 1, 178, 0, 1.2, 1, 0, 3]])

# Scale the features
input_data[:, scaled_features] = sc.transform(input_data[:, scaled_features])

prediction = loaded_model.predict(input_data)
print(prediction)

if (prediction == 1):
  print('The person has heart disease')
else:
  print('The object does not have heart disease')
