from django.shortcuts import render
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'diabetes-1.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
    logger.debug(f"Model loaded: {type(model)}")
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

try:
    std_scaler = joblib.load(SCALER_PATH)
    logger.debug("Scaler loaded")
except FileNotFoundError:
    logger.warning(f"Scaler file not found at {SCALER_PATH}; using new StandardScaler. Predictions may be inaccurate.")
    std_scaler = StandardScaler()

def predict(request):
    if request.method == 'POST':
        try:
            logger.debug("Received POST request")
            # Get raw input from form
            pregnancies = float(request.POST.get('pregnancies', 0))
            glucose = float(request.POST.get('glucose', 0))
            blood_pressure = float(request.POST.get('blood_pressure', 0))
            skin_thickness = float(request.POST.get('skin_thickness', 0))
            insulin = float(request.POST.get('insulin', 0))
            bmi = float(request.POST.get('bmi', 0))
            diabetes_pedigree = float(request.POST.get('diabetes_pedigree', 0))
            age = float(request.POST.get('age', 0))
            logger.debug(f"Inputs: {pregnancies}, {glucose}, {blood_pressure}, {skin_thickness}, {insulin}, {bmi}, {diabetes_pedigree}, {age}")

            # Create DataFrame
            input_data = pd.DataFrame({
                'Pregnancies': [pregnancies],
                'Glucose': [glucose],
                'BloodPressure': [blood_pressure],
                'SkinThickness': [skin_thickness],
                'Insulin': [insulin],
                'BMI': [bmi],
                'DiabetesPedigreeFunction': [diabetes_pedigree],
                'Age': [age]
            })

            # Preprocessing: Replace 0s with NaN
            cols_to_replace = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            input_data[cols_to_replace] = input_data[cols_to_replace].replace(0, np.NaN)

            # Fill NaNs with medians
            for col in cols_to_replace:
                if input_data[col].isna().all():
                    logger.error(f"Invalid input for {col}: all values are NaN")
                    return render(request, 'form.html', {'error': f"Invalid input for {col}: please provide valid values"})
                input_data[col] = input_data[col].fillna(input_data[col].median())
            logger.debug(f"After NaN handling:\n{input_data}")

            # Feature Engineering: NewBMI
            NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"])
            input_data['NewBMI'] = ''
            input_data.loc[input_data["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
            input_data.loc[(input_data["BMI"] >= 18.5) & (input_data["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
            input_data.loc[(input_data["BMI"] > 24.9) & (input_data["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
            input_data.loc[(input_data["BMI"] > 29.9) & (input_data["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
            input_data.loc[(input_data["BMI"] > 34.9) & (input_data["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
            input_data.loc[input_data["BMI"] > 39.9, "NewBMI"] = NewBMI[5]

            # Feature Engineering: NewInsulinScore
            def set_insulin(row):
                return "Normal" if 16 <= row["Insulin"] <= 166 else "Abnormal"
            input_data['NewInsulinScore'] = input_data.apply(set_insulin, axis=1)

            # Feature Engineering: NewGlucose
            NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"])
            input_data['NewGlucose'] = ''
            input_data.loc[input_data["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]
            input_data.loc[(input_data["Glucose"] > 70) & (input_data["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]
            input_data.loc[(input_data["Glucose"] > 99) & (input_data["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]
            input_data.loc[(input_data["Glucose"] > 126), "NewGlucose"] = NewGlucose[3]
            logger.debug(f"After feature engineering:\n{input_data}")

            # One-Hot Encoding
            input_data = pd.get_dummies(input_data, 
                                      columns=["NewBMI", "NewInsulinScore", "NewGlucose"], 
                                      prefix=["NewBMI", "NewInsulinScore", "NewGlucose"])
            input_data[input_data.select_dtypes(bool).columns] = input_data.select_dtypes(bool).astype(int)

            # Rename columns to match model's typos
            input_data.rename(columns={
                'NewGlucose_Secret': 'NewGlucose_Secert',
                'NewInsulinScore_Normal': 'NewInsulinScore_Noraml'
            }, inplace=True)
            logger.debug(f"After renaming columns:\n{input_data.columns.tolist()}")

            # Ensure columns match training data
            expected_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                             'DiabetesPedigreeFunction', 'Age', 'NewBMI_Obesity 1', 'NewBMI_Obesity 2', 
                             'NewBMI_Obesity 3', 'NewBMI_Overweight', 'NewBMI_Underweight', 
                             'NewInsulinScore_Noraml', 'NewGlucose_Low', 'NewGlucose_Normal', 
                             'NewGlucose_Overweight', 'NewGlucose_Secert']
            for col in expected_cols:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[expected_cols]
            logger.debug(f"After column alignment:\n{input_data}")

            # Apply StandardScaler
            input_scaled = std_scaler.transform(input_data) if os.path.exists(SCALER_PATH) else std_scaler.fit_transform(input_data)
            logger.debug(f"After StandardScaler:\n{input_scaled}")

            # Predict
            prediction = model.predict(input_scaled)[0]
            result = "Diabetes" if prediction == 1 else "No Diabetes"
            logger.debug(f"Prediction: {result}")

            return render(request, 'form.html', {'result': result})

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return render(request, 'form.html', {'error': f"Error: {str(e)}"})

    logger.debug("Rendering form for GET request")
    return render(request, 'form.html', {})