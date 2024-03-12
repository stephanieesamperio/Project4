from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import re
from flask_cors import CORS

# Initialize the Flask application and enable CORS for cross-origin requests
app = Flask(__name__, static_folder='.', static_url_path='/predictive_models_demo.html')
CORS(app)

# Load a pre-trained TensorFlow Keras model for text generation
model = load_model('my_model3.h5')

# Define the character set used during the model's training for consistent preprocessing
chars = [
    ' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z', 'µ', '¾', 'à', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è',
    'é', 'ê', 'ë', 'î', 'ö', 'ù', 'û', 'ü'
]
char_to_index = dict((c, i) for i, c in enumerate(chars))
index_to_char = dict((i, c) for i, c in enumerate(chars))
max_length = 150  # Maximum length of sequences used for model input

# Clean and preprocess input text
def clean_text(text):
    """Normalize text by converting to lowercase, removing extra spaces and punctuation."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_input(text):
    """Prepare input text for model prediction, encoding it into a one-hot matrix."""
    text = clean_text(text)
    x = np.zeros((1, max_length, len(chars)), dtype=bool)
    for t, char in enumerate(text[-max_length:]):
        if char in char_to_index:
            x[0, t, char_to_index[char]] = 1
    return x

def sample(preds, temperature=1.0):
    """Sample an index from a probability array, using a specified temperature to affect diversity."""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(input_text, length=100, temperature=1.0):
    """Generate text based on the input text, model predictions, and given parameters."""
    generated = ''
    sentence = input_text[-max_length:]
    generated += sentence

    for i in range(length):
        x_pred = np.zeros((1, max_length, len(chars)))
        for t, char in enumerate(sentence):
            if char in char_to_index:  # Ensure character is in char_to_index
                x_pred[0, t, char_to_index[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

# Define routes for serving the prediction form and assets
@app.route('/')
def serve_prediction_form():
    return send_from_directory('.','predictive_models_demo.html')
@app.route('/styles.css')
def serve_css():
    return send_from_directory('.','styles.css')

# Create a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    """Handle POST requests to generate text based on the provided input, length, and temperature."""
    data = request.get_json(force=True)
    input_text = data['input']
    length = data.get('length', 100)  # Default length to 100 if not specified
    temperature = data.get('temperature', 1.0)  # Default temperature to 1.0 if not specified
    generated_text = generate_text(input_text, length, temperature)
    return jsonify({'generated_text': generated_text})

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)