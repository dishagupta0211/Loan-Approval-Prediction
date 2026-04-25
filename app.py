import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('loan_model.pkl', 'rb'))

# Title
st.title("🏦 Loan Approval Prediction System")
st.markdown("Enter details below to check loan approval status")

# Sidebar
st.sidebar.header("User Input Features")

# Inputs
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
married = st.sidebar.selectbox("Married", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])
credit_history = st.sidebar.selectbox("Credit History", [0, 1])
property_area = st.sidebar.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

loan_term = st.sidebar.number_input("Loan Amount Term")
income_log = st.sidebar.number_input("Applicant Income (log)")
loan_log = st.sidebar.number_input("Loan Amount (log)")

# Convert inputs to numeric
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Not Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

if property_area == "Rural":
    property_area = 0
elif property_area == "Semiurban":
    property_area = 1
else:
    property_area = 2

# Prediction
if st.button("Predict"):
    data = np.array([[gender, married, dependents, education,
                      self_employed, credit_history, property_area,
                      loan_term, income_log, loan_log]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("✅ Loan is likely to be Approved")
    else:
        st.error("❌ Loan is likely to be Rejected")

# About section
st.markdown("""
---
### 📌 About Project
This ML model predicts whether a loan will be approved based on applicant details.  
Model used: Logistic Regression (with class balancing).
""")