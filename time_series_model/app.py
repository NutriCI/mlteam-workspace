import flask
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
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
    required_keys = ['calories', 'sugar', 'fat', 'salt']
    for key in required_keys:
        if key not in data or not isinstance(data[key], list):
            return jsonify({'error': f"Input must contain a list with 7 numbers for {key}."}), 400
    
    try:
        # Make sure each input has length of 7
        input_data = []
        for key in required_keys:
            if len(data[key]) != 7:
                return jsonify({'error': f"Input for {key} must be a list of length 7."}), 400
            input_data.append(data[key])
        
        # Convert to numpy array and reshape
        input_data = np.array(input_data).T.reshape((1, 7, 4))  # (batch_size, timesteps, features)
        
        # Select only the 'calories' feature for prediction
        input_data = input_data[:, :, 0].reshape((1, 7, 1))  # (batch_size, timesteps, 1)
        
        # Predict
        prediction = model.predict(input_data)

        # Format prediction result into JSON
        response = {
            'predicted_calories': int(prediction[0][0])
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask server
if __name__ == '__main__':
    app.run(debug=True)