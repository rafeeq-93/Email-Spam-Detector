from flask import Flask, request, jsonify, render_template
from model import SpamDetector  # Import the SpamDetector class

app = Flask(__name__)
detector = SpamDetector()  # Initialize the SpamDetector

@app.route('/')
def home():
    """Render the home page with the input form."""
    return render_template('spam.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict if an email is spam or not."""
    data = request.get_json()  # Get JSON data from the request
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']  # Extract the text to predict
    prediction = detector.predict(text)  # Get the prediction

    if prediction is None:
        return jsonify({'error': 'Model not loaded, cannot predict'}), 500

    result = 'spam' if prediction == 1 else 'not spam'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)