import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load both the model and label encoders
try:
    with open('random_forest_model.pkl', 'rb') as file:
        rf = pickle.load(file)
    with open('label_encoders.pkl', 'rb') as file:
        le_dict = pickle.load(file)
except FileNotFoundError:
    st.error("Model or encoder files not found. Please ensure both .pkl files exist in the directory.")
    st.stop()

# Title of the app
st.title('Financial Inclusion Predictor')

# Create input fields for the features
st.header('Enter Customer Information')

# Use the actual encoded values from label encoders
education_options = list(le_dict['education_level'].classes_)
education_level = st.selectbox('Education Level', education_options)
education_encoded = le_dict['education_level'].transform([education_level])[0]

age_of_respondent = st.number_input('Age of Respondent', min_value=18, max_value=100)

cellphone_options = list(le_dict['cellphone_access'].classes_)
cellphone_access = st.radio('Has Cellphone Access', cellphone_options)
cellphone_encoded = le_dict['cellphone_access'].transform([cellphone_access])[0]

job_options = list(le_dict['job_type'].classes_)
job_type = st.selectbox('Job Type', job_options)
job_encoded = le_dict['job_type'].transform([job_type])[0]

country_options = list(le_dict['country'].classes_)
country = st.selectbox('Country', country_options)
country_encoded = le_dict['country'].transform([country])[0]

# Create prediction button
if st.button('Predict Bank Account Status'):
    features = np.array([[education_encoded, age_of_respondent, cellphone_encoded,
                         job_encoded, country_encoded]])
    try:
        prediction = rf.predict(features)
        
        st.subheader('Prediction Result:')
        if prediction[0] == 1:
            st.success('This person is likely to have a bank account.')
        else:
            st.error('This person is unlikely to have a bank account.')

        proba = rf.predict_proba(features)
        st.write(f'Probability of having a bank account: {proba[0][1]:.2%}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add some information about the model
st.sidebar.header('About this Model')
st.sidebar.write('''
This model predicts whether an individual is likely to have a bank account based on:
- Education Level
- Age
- Cellphone Access
- Job Type
- Country of Residence

The model uses a Random Forest Classifier trained on financial inclusion data.
''')

# Add footer
st.markdown('''
---
Created for Financial Inclusion Prediction
''')