import joblib
import pandas as pd
import numpy as np

# Carregar o modelo
model = joblib.load('model/model.pkl')

# Nova entrada para previsão
input_data = pd.DataFrame({
    "Pregnancies": [1],
    "PlasmaGlucose": [85],
    "DiastolicBloodPressure": [66],
    "TricepsThickness": [29],
    "SerumInsulin": [0],
    "BMI": [26.6],
    "DiabetesPedigree": [0.351],
    "Age": [34]
})

# Realizar previsão
prediction = model.predict(input_data)
print(f'Prediction: {prediction}')
