from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import joblib
from dotenv import load_dotenv
import logging
from LogisticRegresson import LogisticRegression

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load model from local file using joblib
try:
    logger.info("Loading local model using joblib: model.pkl")
    model = joblib.load("logistic_model.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load local model: {str(e)}")
    raise RuntimeError("Model loading failed")

# Load preprocessing artifacts (encoder & scaler)
try:
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    logger.info("Preprocessing artifacts loaded successfully")
except Exception as e:
    logger.error(f"Failed to load preprocessing artifacts: {str(e)}")
    raise RuntimeError("Preprocessing artifacts loading failed")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        fuel = request.form['fuel']
        owner = request.form['owner']
        brand = request.form['brand']
        km_driven = float(request.form['km_driven'])
        seats = int(request.form['seats'])
        year = int(request.form['year'])
        engine = int(request.form['engine'])

        # Prepare input data
        input_data = pd.DataFrame({
            'fuel': [fuel],
            'owner': [owner],
            'brand': [brand],
            'km_driven': [km_driven],
            'engine': [engine],
            'seats': [seats],
            'year': [year]
        })

        # Preprocess data
        categorical_cols = ['fuel', 'owner', 'brand']
        encoded_categorical = encoder.transform(input_data[categorical_cols])
        
        numerical_cols = ['km_driven', 'engine']
        scaled_numerical = scaler.transform(input_data[numerical_cols])
        
        other_features = input_data[['seats', 'year']].values

        # Combine features
        features = np.hstack([
            scaled_numerical,
            encoded_categorical,
            other_features
        ])

        # Pad with zeros if needed
        if features.shape[1] < 40:
            padding = np.zeros((features.shape[0], 40 - features.shape[1]))
            features = np.hstack([features, padding])

        logger.info(f"Final input shape: {features.shape}")

        # Predict
        prediction = int(model.predict(features)[0])
        category_labels = {0: "0:Budget", 1: "1 :Mid-Range", 2: "3:Premium"}

        return render_template(
            'result.html',
            prediction=f"Predicted Price Category: {category_labels.get(prediction, 'Unknown')}"
        )

    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        return render_template('index.html', error=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', error="An error occurred. Please try again.")

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
