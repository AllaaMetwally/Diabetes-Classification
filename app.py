import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import pickle
import os


@st.cache_resource
def load_model():
    # هنا بنفترض إنك حفظتي الموديل والسكيلر باستخدام pickle
    # لو مش حابة تستخدمي ملفات خارجية، تقدري تحطي كود التدريب هنا
    with open('svm_model.sav', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.sav', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# عنوان الصفحة بتنسيق شيك
st.title("🩺 Diabetes Prediction Web App")
st.write("This app uses Machine Learning (Support vector machines) to predict diabetes based on medical data.")

# 2. إنشاء المدخلات في الـ Sidebar أو الصفحة الرئيسية
st.header("Patient Health Data")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=158)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=30)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=328)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=35.5)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.344)
    age = st.number_input("Age", min_value=0, max_value=120, value=35)

# 3. زر التنبؤ
if st.button("Predict Results"):
    # تجهيز البيانات كما فعلنا في الكود السابق
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    input_array = np.asarray(input_data).reshape(1, -1)
    
    # خطوة الـ Scaling (حرجة جداً للـ SVM)
    # ملاحظة: استخدمي السكيلر 'scl' اللي دربناه
    std_data = scl.transform(input_array)
    
    # التنبؤ باستخدام SVM
    prediction = svm_model.predict(std_data)
    
    st.markdown("---")
    if prediction[0] == 0:
        st.success('✅ The person does not have diabetes (Model: SVM)')
    else:
        st.error('⚠️ The person has diabetes (Model: SVM)')




if os.path.exists('scaler.sav') and os.path.exists('svm_model.sav'):
    scl = pickle.load(open('scaler.sav', 'rb'))
    svm_model = pickle.load(open('svm_model.sav', 'rb'))
else:
    st.error("⚠️ ملفات الموديل (scaler.sav أو svm_model.sav) غير موجودة. تأكدي من رفعها على GitHub.")
    st.stop()