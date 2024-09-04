import streamlit as st
import numpy as np
import pickle

# Load the models and scalers
model = pickle.load(open('model/gnb_model.pkl', 'rb'))
sc = pickle.load(open('model/standardscaler.pkl', 'rb'))
#mx = pickle.load(open('model/minmaxscaler.pkl', 'rb'))

# Crop dictionary for prediction interpretation
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Set up the Streamlit app
st.title("Crop Prediction Application")
st.markdown("### Enter the details below to predict the best crop for cultivation")

# Input fields
N = st.number_input("Nitrogen Content", min_value=0.0, format="%.6f")
P = st.number_input("Phosphorus Content", min_value=0.0, format="%.6f")
K = st.number_input("Potassium Content", min_value=0.0, format="%.6f")
temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, format="%.6f")
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, format="%.6f")
ph = st.number_input("Soil pH Level", min_value=0.0, max_value=14.0, format="%.6f")
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, format="%.6f")

# Predict button
if st.button("Predict Best Crop"):
    # Prepare the feature list and reshape for prediction
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    
    # Scale the features using MinMaxScaler and StandardScaler
    #mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(single_pred)
    
    # Make the prediction
    prediction = model.predict(sc_mx_features)
    
    # Determine the crop and display the result
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"**{crop}** is the best crop to be cultivated right there."
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    
    # Display the result
    st.success(result)
