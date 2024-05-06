import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from PIL import Image
import requests

st.set_page_config(
    page_title="Crop Recommender",
    page_icon="🌿",
    layout='wide',
    initial_sidebar_state="collapsed"
)

def load_model(model_name):
    model_url = f'https://raw.githubusercontent.com/yashps7/FEYNN_Labs_Final_Project/main/model.pkl'
    response = requests.get(model_url)
    if response.status_code == 200:
        # Save the model to a local file
        with open(model_name, 'wb') as f:
            f.write(response.content)
        # Load the model from the local file
        loaded_model = pickle.load(open(model_name, 'rb'))
        return loaded_model
    else:
        print(f"Failed to download the model from {model_url}.")
        return None

def main():
    # Title
    st.markdown(
        """
        <div style="background-color:MEDIUMSEAGREEN;padding:10px;border-radius:10px; font-family: sans-serif; font-size: 2em;">
        <h1 style="color:white;text-align:center;">Crop Recommendation 🌱</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.text("")
    st.write("""
            Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes. Precision agriculture systems aren't all created equal.
            However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.
            """)
    
    st.markdown("---")

    st.header("Finding out the most suitable crop to grow in your farm 👨‍🌾")

    col1, col2 = st.columns([2, 2])
    
    with col1:
        st.subheader("Information of your Soil: ")
        N = st.slider("Nitrogen", 1, 250, 10)
        P = st.slider("Phosphorus", 1, 250, 10)
        K = st.slider("Potassium", 1, 250, 10)
        
    with col2:
        st.subheader("Predicted Conditions this year: ")

        temp = st.slider("Temperature (°C)", 0.0, 50.0, 10.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 10.0)
        ph = st.slider("pH", 3.0, 11.0, 3.0)
        rainfall = st.slider("Rainfall (mm)", 0.0, 1000.0, 50.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

    if st.button('Predict'):
        loaded_model = load_model('model.pkl')
        prediction = loaded_model.predict(single_pred)
        st.write('## Results 🔍')
        st.success(f"{prediction.item().title()} are recommended by the A.I. for your this season's Agriculture.")

    st.warning("This A.I. application is for demo purposes only and cannot be relied upon.")

if __name__ == '__main__':
    main()
