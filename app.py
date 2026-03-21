import os
import io
import base64
import random
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Load model
try:
    model = tf.keras.models.load_model('model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class names for FashionMNIST
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load test dataset for the random image feature
(_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_random_image', methods=['GET'])
def get_random_image():
    idx = random.randint(0, len(x_test) - 1)
    img_array = x_test[idx]
    label = CLASS_NAMES[y_test[idx]]
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return jsonify({
        'image': f'data:image/png;base64,{img_str}',
        'label': label,
        'index': idx
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read image
        img = Image.open(file.stream).convert('L') # Convert to grayscale
        img = img.resize((28, 28)) # Resize to 28x28
        
        # Preprocess
        img_array = np.array(img)
        # If the image has white background (meaning > 127 mean), invert it since FashionMNIST is black background
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
            
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Predict
        prediction = model.predict(img_array)
        class_idx = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        
        return jsonify({
            'class': CLASS_NAMES[class_idx],
            'confidence': f'{confidence*100:.2f}%',
            'probabilities': {CLASS_NAMES[i]: float(prediction[0][i]) for i in range(10)}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
