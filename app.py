import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("voting_classifier_model.pkl")

# App title
st.title("Deployment")
st.write("This app predicts whether a client will subscribe to a term deposit using a trained Voting Classifier.")

# Input fields for features (Example: Replace with your feature names)
st.header("Input Features")
age = st.number_input("Age", min_value=18, max_value=100, step=1)
duration = st.number_input("Duration (seconds)", min_value=0, step=1)
campaign = st.number_input("Number of Campaign Contacts", min_value=0, step=1)
pdays = st.number_input("Days Since Last Contact", min_value=-1, step=1)
previous = st.number_input("Number of Previous Contacts", min_value=0, step=1)
emp_var_rate = st.number_input("Employment Variation Rate", step=0.1)
cons_price_idx = st.number_input("Consumer Price Index", step=0.01)
cons_conf_idx = st.number_input("Consumer Confidence Index", step=0.1)
euribor3m = st.number_input("Euribor 3-Month Rate", step=0.01)
nr_employed = st.number_input("Number of Employees", step=0.01)

# Collect input features
input_features = np.array([[age, duration, campaign, pdays, previous, emp_var_rate,
                            cons_price_idx, cons_conf_idx, euribor3m, nr_employed]])

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_features)[0]
    probabilities = model.predict_proba(input_features)[0]

    # Display results
    st.subheader("Prediction")
    st.write(f"The predicted class is: **{'Subscribed' if prediction == 1 else 'Not Subscribed'}**")

    st.subheader("Probabilities")
    st.write(f"Probability of Not Subscribed: **{probabilities[0]:.2f}**")
    st.write(f"Probability of Subscribed: **{probabilities[1]:.2f}**")
