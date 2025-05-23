import numpy as np
import tensorflow as tf
import joblib

# Load the model, scaler, and label encoder
model = tf.keras.models.load_model('crop_environment_model.h5')
scaler = joblib.load('environment_scaler.pkl')
label_encoder = joblib.load('crop_label_encoder.pkl')

# Function to make predictions
def predict_environment(crop_name):
    # Encode crop name
    crop_encoded = label_encoder.transform([crop_name])
    
    # Create feature array
    features = np.array(crop_encoded).reshape(1, -1)
    
    # Make prediction
    prediction_scaled = model.predict(features)
    
    # Inverse transform the scaled prediction
    prediction = scaler.inverse_transform(prediction_scaled)
    
    return prediction[0]

# Continuous prediction loop
while True:
    # User input
    crop_name = input("Enter crop name (or type 'exit' to stop): ")
    
    # Check if the user wants to exit
    if crop_name.lower() == 'exit':
        print("Exiting the program.")
        break
    
    # Predict and display the result
    try:
        environment_conditions = predict_environment(crop_name)
        print(f"The predicted environmental conditions for '{crop_name}' are:")
        print(f"N: {environment_conditions[0]}")
        print(f"P: {environment_conditions[1]}")
        print(f"K: {environment_conditions[2]}")
        print(f"Temperature: {environment_conditions[3]}")
        print(f"Humidity: {environment_conditions[4]}")
        print(f"pH: {environment_conditions[5]}")
        print(f"Rainfall: {environment_conditions[6]}")
    except Exception as e:
        print(f"Error: {e}. Please check the crop name or the model.")
