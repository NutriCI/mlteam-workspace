import flask
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

# Load model
model = tf.keras.models.load_model('model/calories_fitbit_model_optmz.h5')

# Flask initialization
app = Flask(__name__)

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from JSON input
    data = request.get_json()

    # Validate input
    if "calories" not in data or not isinstance(data['calories'], list):
        return jsonify({'error': "Input harus berisi list dengan 7 angka."}), 400
    
    try:
        # Pastikan input memiliki panjang 7
        input_data = np.array(data['calories'])
        if len(input_data) != 7:
            return jsonify({'error': "Input harus berupa list dengan panjang 7."}), 400

        # Reshape data menjadi (1, 7, 1) -> (batch_size, timesteps, features)
        input_data = input_data.reshape((1, 7, 1))
        
        # Predict
        prediction = model.predict(input_data)

        # Format prediction result into JSON
        response = {
            'predicted_calories': float(prediction[0][0])
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Jalankan server Flask
if __name__ == '__main__':
    app.run(debug=True)
