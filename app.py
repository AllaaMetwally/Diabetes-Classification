import streamlit as st
import numpy as np
import pickle

# تحميل الموديل والـ scaler اللي حفظناهم
model = pickle.load(open('trained_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

def diabetes_prediction(input_data):
    # تحويل البيانات لـ numpy array وعمل reshape
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # عمل scaling للبيانات
    std_data = scaler.transform(input_data_reshaped)
    
    prediction = model.predict(std_data)
    
    if (prediction[0] == 0):
        return 'The person does not have diabetes ✅'
    else:
        return 'The person has diabetes ⚠️'

def main():
    # عنوان الصفحة
    st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
    st.title('Diabetes Prediction Web App 🩺')
    st.write('Enter the following details to check the prediction:')

    # تقسيم المدخلات لصفوف عشان الشكل يكون أحسن
    col1, col2 = st.columns(2)
    
    with col1:
        Pregnancies = st.number_input('Pregnancies', min_value=0, step=1)
        BloodPressure = st.number_input('Blood Pressure ', min_value=0)
        Insulin = st.number_input('Insulin Level', min_value=0)
        Age = st.number_input('Age', min_value=0)

    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0)
        SkinThickness = st.number_input('Skin Thickness ', min_value=0)
        BMI = st.number_input('BMI (Body Mass Index)', format="%.1f")
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function ', format="%.3f")

    # زرار التوقع
    diagnosis = ''
    if st.button('Predict'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    if diagnosis:
        if 'not' in diagnosis:
            st.success(diagnosis)
        else:
            st.error(diagnosis)

if __name__ == '__main__':
    main()