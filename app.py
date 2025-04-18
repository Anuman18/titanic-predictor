import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.title("🚢 Titanic Survival Predictor")

# Sidebar input
st.sidebar.header("Passenger Info")

Pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Sex = st.sidebar.selectbox("Sex (0 = Male, 1 = Female)", [0, 1])
Age = st.sidebar.slider("Age", 0, 100, 30)
SibSp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.sidebar.slider("Parents/Children Aboard", 0, 10, 0)
Fare = st.sidebar.slider("Fare", 0, 500, 50)
Embarked = st.sidebar.selectbox("Embarked (0 = C, 1 = Q, 2 = S)", [0, 1, 2])

# Prediction button
if st.sidebar.button("Predict"):
    # Create feature array
    input_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[0]

    # Output
    if prediction[0] == 1:
        st.success(f"✅ The passenger **survived**! (Probability: {prediction_proba[1]:.2f})")
    else:
        st.error(f"❌ The passenger **did not survive**. (Probability: {prediction_proba[0]:.2f})")
