import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data from CSV
df = pd.read_csv(r"D:\csv_files\new_crop_data.csv")

# Split features and labels
X = df[['label']]
y = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(X['label'])

# Convert labels to DataFrame
y_encoded_df = pd.DataFrame(y_encoded, columns=['label'])

# Combine features and encoded labels
data = pd.concat([y_encoded_df, y], axis=1)

# Split data into features and labels
X_train, X_test, y_train, y_test = train_test_split(data[['label']], data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']], test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y_train_scaled.shape[1])
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train_scaled, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test_scaled)
print(f'Test loss: {loss:.4f}')

# Save the model, scaler, and label encoder
model.save('crop_environment_model.h5')
import joblib
joblib.dump(scaler, 'environment_scaler.pkl')
joblib.dump(label_encoder, 'crop_label_encoder.pkl')
