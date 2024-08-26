import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as p
import pickle

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Load all the scaler, one hot encoder, Label Encoder
with open("o.pkl", "rb") as file:
    o = pickle.load(file)

with open("l.pkl", "rb") as file:
    l = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit App
st.title("BAKER'S CHURN PREDICTION")

# User Input
geography = st.selectbox("Geography", o.categories_[0])
gender = st.selectbox("Gender", l.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Create input DataFrame
input_data = p.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [l.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# OneHot Encode Geography
geo_encoded = o.transform([[geography]]).toarray()
geo_feature_names = o.get_feature_names_out(["Geography"])
geo_encoded_df = p.DataFrame(geo_encoded, columns=geo_feature_names)

# Combine the onehot encoded geography with other features
input_data = p.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_scaled)
prediction_probability = prediction[0][0]

# Display prediction result
if prediction_probability > 0.5:
    st.write("Most Probably will terminate the account")
else:
    st.write("Most Probably will not terminate the account")

# Display prediction probability
st.write(f"Prediction Probability: {prediction_probability}")