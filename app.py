import streamlit as st
import pandas as pd
import joblib
from src.preprocess import preprocess_data

st.title("Personal Loan Approval Predictor")

model = joblib.load("loan_model.pkl")

# Input fields
age = st.number_input("Age", 18, 100)
income = st.number_input("Annual Income", 1000, 1000000)
credit_score = st.number_input("Credit Score", 300, 850)
gender = st.selectbox("Gender", ["Male", "Female"])

# Make prediction
input_df = pd.DataFrame({
    "Age":[age],
    "Income":[income],
    "CreditScore":[credit_score],
    "Gender":[gender]
})

X, _ = preprocess_data(pd.concat([input_df, pd.DataFrame({"LoanApproved":[0]})], axis=1))
prediction = model.predict(X)

st.subheader("Prediction")
st.write("Loan Approved ✅" if prediction[0]==1 else "Loan Denied ❌")
