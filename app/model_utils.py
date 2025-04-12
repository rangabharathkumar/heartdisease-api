import joblib
import numpy as np

# Load the model artifacts from the artifacts folder
# Adjust paths as needed (here we assume the working directory is the project root)
model = joblib.load("app/artifacts/heartdisease_model.pkl")
scaler = joblib.load("app/artifacts/scaler.pkl")

# If your training pipeline used label encoders for categorical columns, load them; otherwise, omit this.
try:
    label_encoders = joblib.load("app/artifacts/label_encoders.pkl")
except Exception:
    label_encoders = None

def preprocess_input(input_data: dict) -> np.ndarray:
    # For example, if you encoded categorical features manually, you would transform them here
    # If label_encoders is not None, you might do something like:
    if label_encoders:
        input_data['cp'] = label_encoders['cp'].transform([input_data['cp']])[0]
        # Repeat for any other features you encoded during training

    # Arrange the input features in the same order as the training data
    feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 
                     'fbs', 'restecg', 'thalachh', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    features = [input_data[feature] for feature in feature_order]
    features = np.array([features])

    # Apply the scaler for numerical features
    features_scaled = scaler.transform(features)
    return features_scaled

def predict_heartdisease(input_data: dict):
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    return int(prediction[0])  # or decode the class label if needed
