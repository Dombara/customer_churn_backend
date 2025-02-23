from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import os
from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["POST", "OPTIONS", "GET"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

model = tf.keras.models.load_model("customer_churn_hackathon.h5")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

scaler = joblib.load('scaler.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Hello World", "status": 200})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read and preprocess the data
        data = pd.read_csv(request.files['file'])
        original_data = data.copy()  # Keep a copy of original data
        
        X = data.iloc[:, 3:-1].values
        X[:, 2] = label_encoder.transform(X[:, 2])
        X = np.array(onehot_encoder.transform(X))
        X = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Get high risk customers
        high_risk_indices = np.where(predictions > 0.5)[0]
        high_risk_customers = []
        
        # For each high risk customer, store their original data
        for idx in high_risk_indices:
            customer_data = original_data.iloc[idx, 3:-1].to_dict()
            customer_data['churn_probability'] = float(predictions[idx])
            high_risk_customers.append(customer_data)

        return jsonify({
            "predictions": high_risk_customers,
            "status": 200
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', debug=True, port=port)