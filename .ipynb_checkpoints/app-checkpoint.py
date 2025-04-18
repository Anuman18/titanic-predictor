import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model using joblib
model = joblib.load("titanic_model.pkl")  # Load the saved model

# Function to make predictions
def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    input_data = np.array([Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]).reshape(1, -1)
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Streamlit interface
st.title("Titanic Survival Predictor")
st.sidebar.header("Input Features")

# Example Input Values (Passenger who survived)
Pclass = 1  # First class
Sex = 1  # Female
Age = 30  # 30 years old
SibSp = 0  # No siblings or spouse aboard
Parch = 0  # No parents or children aboard
Fare = 50  # Fare paid
Embarked = 0  # Embarked from Cherbourg

# Display the input values on the Streamlit app
st.sidebar.write(f"Pclass: {Pclass}")
st.sidebar.write(f"Sex: {Sex}")
st.sidebar.write(f"Age: {Age}")
st.sidebar.write(f"SibSp: {SibSp}")
st.sidebar.write(f"Parch: {Parch}")
st.sidebar.write(f"Fare: {Fare}")
st.sidebar.write(f"Embarked: {Embarked}")

if st.sidebar.button("Predict"):
    survival = predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
    if survival == 1:
        st.success("The passenger survived!")
    else:
        st.error("The passenger did not survive.")
