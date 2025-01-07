import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model and scaler
model = joblib.load("penguin_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title('🐧 Penguin Species Prediction')
st.write('Enter the penguin characteristics below to predict its species.')

# Numerical input for features
bill_length_mm = st.number_input('Bill Length (mm)', value=45.0, step=0.1)
bill_depth_mm = st.number_input('Bill Depth (mm)', value=15.0, step=0.1)
flipper_length_mm = st.number_input('Flipper Length (mm)', value=200.0, step=1.0)
body_mass_g = st.number_input('Body Mass (g)', value=4000.0, step=1.0)
sex = st.radio('Sex', ['Male', 'Female'])
island = st.radio('Island', ['Biscoe', 'Torgersen', 'Dream'])

if st.button('Predict', type='primary'):
    # Prepare input data
    # Numeric features
    numeric_data = pd.DataFrame({
        'bill_length_mm': [bill_length_mm],
        'bill_depth_mm': [bill_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g]
    })

    # Scale numeric features
    numeric_data_scaled = scaler.transform(numeric_data)

    # Categorical features (one-hot encoding)
    categorical_data = pd.DataFrame({
        'sex': [1 if sex == 'Male' else 0],
        'island_Dream': [1 if island == 'Dream' else 0],
        'island_Torgersen': [1 if island == 'Torgersen' else 0]
    })

    # Combine scaled numeric data with categorical data
    input_data = pd.DataFrame(numeric_data_scaled, columns=numeric_data.columns)
    input_data = pd.concat([input_data, categorical_data], axis=1)

    # Ensure feature order matches training data
    feature_order = [
        'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 
        'sex', 'island_Dream', 'island_Torgersen'
    ]
    input_data = input_data[feature_order]

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Show results
    st.success(f'Predicted Species: {prediction}')
    st.write('Prediction Probabilities:')

    for species, prob in zip(model.classes_, probability):
        st.progress(prob)
        st.write(f'{species}: {prob:.2%}')

    # Display user input
    st.subheader('User Input Features')
    st.write(input_data)
