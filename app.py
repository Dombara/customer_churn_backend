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
# from flask_pymongo import PyMongo
from pymongo import MongoClient


app = Flask(__name__)

# app.config["MONGO_URI"] = "mongodb+srv://yash:yash9999@cluster0.glzt8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# MONGO_URI="mongodb+srv://yash:yash9999@cluster0.glzt8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient("mongodb+srv://yash:yash9999@cluster0.glzt8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["churndb"]
# collection = db["data"]



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

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Read and preprocess the data
#         data = pd.read_csv(request.files['file'])
#         original_data = data.copy()  # Keep a copy of original data
        
#         X = data.iloc[:, 3:-1].values
#         X[:, 2] = label_encoder.transform(X[:, 2])
#         X = np.array(onehot_encoder.transform(X))
#         X = scaler.transform(X)
        
#         # Make predictions
#         predictions = model.predict(X)
        
#         # Get high risk customers
#         high_risk_indices = np.where(predictions > 0.5)[0]
#         high_risk_customers = []
        
#         # For each high risk customer, store their original data
#         for idx in high_risk_indices:
#             customer_data = original_data.iloc[idx].to_dict()
#             customer_data['churn_probability'] = float(predictions[idx])
#             high_risk_customers.append(customer_data)

#         return jsonify({
#             "predictions": high_risk_customers,
#             "status": 200
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)})




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
        
        # Store all customer data with their churn probabilities
        all_customers = []
        
        for idx in range(len(original_data)):  # Iterate over all customers
            customer_data = original_data.iloc[idx].to_dict()
            customer_data['churn_probability'] = float(predictions[idx])  # Add churn probability
            all_customers.append(customer_data)

        return jsonify({
            "predictions": all_customers,
            "status": 200
        })

    except Exception as e:
        return jsonify({"error": str(e)})



@app.route('/save-predictions', methods=['POST'])
def save_predictions():
    try:
        data = request.json  # Get JSON data from request
        # predictions = data.get('datasetName', [])
        collection_name=data.get('datasetName')
        if not collection_name:
            return jsonify({"message": "Collection name is required"}), 400


        predictions=data.get('predictions', [])

        if not predictions:
            return jsonify({"message": "No predictions provided"}), 400

        collection=db[collection_name]

        # # Insert predictions into MongoDB
        result = collection.insert_many(predictions)

        return jsonify({"message": "Predictions saved successfully"}),200
    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route('/get-predictions-data', methods=['POST'])
def get_predictions():
    try:
        print("req hit: ", request.json)
        data = request.json  # Get JSON data from request
        collection_name=data.get('collectionName')
        print(collection_name)

        # print("hello yash11")
        if not collection_name:
            return jsonify({"message": "Collection name is required"}), 400
        # print("hello yash")
        
        collection = db[collection_name]
        # print("hello yash")
        print(db[collection_name])
         # Convert ObjectId to string for each document
        predictions = list(collection.find())
        for doc in predictions:
            doc['_id'] = str(doc['_id'])  # Convert ObjectId to string

        # predictions = list(collection.find({}, {"_id": 0}))

        return jsonify({
            "predictions": predictions,
            "status": 200
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500





@app.route('/get-titles', methods=['GET'])
def get_titles():
    try:
        collection_names = db.list_collection_names()
        return jsonify({
            "collections": collection_names,
            "status": 200
        })
    except Exception as e: 
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)