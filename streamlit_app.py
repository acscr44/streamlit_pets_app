import streamlit as st
import pandas as pd
import joblib

# Title
st.title("App con Modelo de Predicción de Mascota")
st.write("")

# Add a text input
#name = st.text_input("Enter your name:")

# Input bar 1
height = st.number_input('Insert Height value')

# Input bar 2
weight = st.number_input('Insert Weight value')

# Dropdown input
eyes = st.selectbox("Select Eye Colour", ("Blue", "Brown"))
#option = st.select_slider('Select an option', options=['Perro', 'Gato'])

# If button is pressed
if st.button("Submit"):

    # Unpickle classifier
    pet_model = joblib.load("pet_model.pkl")

    # Store inputs into dataframe
    X = pd.DataFrame([[height, weight, eyes]], columns=["Height", "Weight", "Eye"])
    X = X.replace(["Brown", "Blue"], [1, 0])

    # Get prediction
    prediction = pet_model.predict(X)[0]

    # Output prediction
    st.text(f"This instance is a {prediction}")