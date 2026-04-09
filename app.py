import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Impact Predictor", page_icon="🤖", layout="centered")

# --- CUSTOM CSS FOR ANIMATION & STYLE ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_m9p8llmj.json")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open("Model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- APP UI ---
st.title("🤖 AI Impact Analysis Tool")
st.subheader("Predicting the influence of AI on Academic Performance")

if lottie_ai:
    st_lottie(lottie_ai, height=200, key="coding")

st.write("---")

# Based on your model's feature_names_in_: 
# ['Age', 'Gender', 'Education_Level', 'City', 'AI_Tool_Used', 'Daily_Usage_Hours', 'Purpose', 'Impact_on_Grades']
# Note: Since KNN requires numbers, ensure these match the encoding used during training.

st.write("### 📝 Enter Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x==0 else "Female")
    edu = st.selectbox("Education Level", options=[0, 1, 2], format_func=lambda x: ["School", "Undergrad", "Postgrad"][x])
    city = st.number_input("City Code", min_value=0, step=1)

with col2:
    ai_tool = st.selectbox("Primary AI Tool", options=[0, 1, 2], format_func=lambda x: ["ChatGPT", "Gemini", "Others"][x])
    usage = st.slider("Daily Usage (Hours)", 0.0, 24.0, 2.0)
    purpose = st.selectbox("Purpose", options=[0, 1], format_func=lambda x: "Education" if x==0 else "Personal")
    current_impact = st.number_input("Current Impact Score", value=0.0)

# --- PREDICTION LOGIC ---
if st.button("Analyze Impact"):
    # Create input array
    features = np.array([[age, gender, edu, city, ai_tool, usage, purpose, current_impact]])
    
    prediction = model.predict(features)
    
    st.write("---")
    st.balloons()
    
    # Display Result
    st.success(f"### Prediction Result: {prediction[0]}")
    st.info("The model classifies the outcome based on the K-Nearest Neighbors algorithm.")

st.markdown("---")
st.caption("Powered by Scikit-Learn 1.6.1 and Streamlit")
