import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
import numpy as np

# Define the PyTorch model class
class PenguinModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PenguinModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)


# Load pre-trained models and scaler
try:
    sklearn_model = joblib.load("penguin_model.pkl")  # Replace with actual scikit-learn model file path
except FileNotFoundError as e:
    st.error(f"Error loading the scikit-learn model: {e}")
    
try:
    scaler = joblib.load("scaler.pkl")  # Replace with actual scaler file path
except FileNotFoundError as e:
    st.error(f"Error loading the scaler: {e}")

# Load the PyTorch model
try:
    pytorch_model = PenguinModel(input_size=7, output_size=3)
    pytorch_model.load_state_dict(torch.load("penguin_pytorch_model.pth"))  # Replace with PyTorch model path
    pytorch_model.eval()
except Exception as e:
    st.error(f"Error loading the PyTorch model: {e}")

# Streamlit app interface
st.title('🐧 Penguin Species Prediction')
st.write('Enter the penguin characteristics below to predict its species using your preferred model.')

# User inputs
bill_length_mm = st.number_input('Bill Length (mm)', value=45.0, step=0.1)
bill_depth_mm = st.number_input('Bill Depth (mm)', value=15.0, step=0.1)
flipper_length_mm = st.number_input('Flipper Length (mm)', value=200.0, step=1.0)
body_mass_g = st.number_input('Body Mass (g)', value=4000.0, step=1.0)
sex = st.radio('Sex', ['Male', 'Female'])
island = st.radio('Island', ['Biscoe', 'Torgersen', 'Dream'])
model_choice = st.radio('Choose Model', ['scikit-learn', 'PyTorch'])

if st.button('Predict', type='primary'):
    # Prepare input data
    numeric_data = pd.DataFrame({
        'bill_length_mm': [bill_length_mm],
        'bill_depth_mm': [bill_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g]
    })
    numeric_data_scaled = scaler.transform(numeric_data)
    categorical_data = pd.DataFrame({
        'sex': [1 if sex == 'Male' else 0],
        'island_Dream': [1 if island == 'Dream' else 0],
        'island_Torgersen': [1 if island == 'Torgersen' else 0]
    })
    input_data = pd.DataFrame(numeric_data_scaled, columns=numeric_data.columns)
    input_data = pd.concat([input_data, categorical_data], axis=1)

    # Ensure feature order
    feature_order = [
        'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 
        'sex', 'island_Dream', 'island_Torgersen'
    ]
    input_data = input_data[feature_order]

    if model_choice == 'scikit-learn':
        # scikit-learn prediction
        prediction = sklearn_model.predict(input_data)[0]
        probability = sklearn_model.predict_proba(input_data)[0]
    else:
        # PyTorch prediction
        input_tensor = torch.tensor(input_data.values, dtype=torch.float32)
        with torch.no_grad():
            output = pytorch_model(input_tensor)
            probability = output.numpy()[0]
            prediction = np.argmax(probability)

    # Map the prediction to the species names
    species_mapping = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}  # Assuming these species
    st.success(f'Predicted Species: {species_mapping.get(prediction, "Unknown Species")}')
    
    # Show prediction probabilities
    st.write('Prediction Probabilities:')
    for species, prob in zip(species_mapping.values(), probability):
        st.progress(float(prob))  # Convert to Python float for progress bar
        st.write(f'{species}: {float(prob):.2%}')

    # Display user input
    st.subheader('User Input Features')
    st.write(input_data)
