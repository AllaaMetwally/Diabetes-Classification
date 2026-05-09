import streamlit as st
import pandas as pd
import numpy as np
import pickle

# إعدادات الصفحة
st.set_page_config(page_title="Diabetes Prediction System", layout="centered")

# إضافة عنوان وشرح
st.title("🩺 Diabetes Prediction Web App")
st.write("This app uses Machine Learning (Random Forest) to predict diabetes based on medical data.")

# دالة لتحميل الموديل والـ Scaler المحفوظين
@st.cache_resource
def load_assets():
    with open('best_model.sav', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.sav', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

try:
    model, scaler = load_assets()

    # تصميم واجهة الإدخال في أعمدة
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=250, value=100)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    # زر التنبؤ
    if st.button("Predict Result"):
        # تجهيز البيانات
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        std_features = scaler.transform(features)
        
        # التنبؤ
        prediction = model.predict(std_features)
        
        st.subheader("Result:")
        if prediction[0] == 1:
            st.error("⚠️ The model predicts that the patient is **Diabetic**.")
        else:
            st.success("✅ The model predicts that the patient is **Non-Diabetic**.")

except FileNotFoundError:
    st.error("Please make sure 'best_model.sav' and 'scaler.sav' are in the same directory.")