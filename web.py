from __future__ import division, print_function  

import os
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Flask app setup
app = Flask(__name__)
CORS(app)

# Load trained model
MODEL_PATH = 'trained_model.h5'

# Define the model architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), name='conv2d_1'),
    MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'),
    Flatten(name='flatten_1'),
    Dense(128, activation='relu', name='dense_1'),
    Dense(128, activation='relu', name='dense_2'),
    Dense(128, activation='relu', name='dense_3'),
    Dense(128, activation='relu', name='dense_4'),
    Dense(13, activation='softmax', name='dense_5')  # 13 output classes
])

# Load the trained weights
model.load_weights(MODEL_PATH)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Class labels
index = ['Basalt', 'Conglomerate', 'Dolostone', 'Gabbro', 'Gneiss', 'Granite',
         'Limestone', 'Marble', 'Quartzite', 'Rhyolite', 'Sandstone', 'Shale', 'Slate']

rock_info = {
    'Basalt': 'an igneous rock.',
    'Conglomerate': 'a sedimentary rock.',
    'Dolostone': 'a sedimentary rock.',
    'Gabbro': 'an igneous rock.',
    'Gneiss': 'a metamorphic rock.',
    'Granite': 'an igneous rock.',
    'Limestone': 'a sedimentary rock.',
    'Marble': 'a metamorphic rock.',
    'Quartzite': 'a metamorphic rock.',
    'Rhyolite': 'an igneous rock.',
    'Sandstone': 'a sedimentary rock.',
    'Shale': 'a sedimentary rock.',
    'Slate': 'a metamorphic rock.'
}

# Flask Routes
@app.route('/', methods=["GET"])
def index():
    return render_template("base.html")

@app.route('/predict', methods=['POST'])
def upload():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        f = request.files['image']
        if f.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save uploaded file
        basepath = os.path.dirname(__file__)
        upload_dir = os.path.join(basepath, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)  # Ensure the directory exists
        file_path = os.path.join(upload_dir, secure_filename(f.filename))
        f.save(file_path)

        # Preprocess the image
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Make prediction
        prediction = model.predict(x)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)  # Get confidence score
        
        # Threshold to detect non-rock images
        threshold = 0.6  
        if confidence < threshold:
            return jsonify({'error': 'Uploaded image is not recognized as a rock. Please upload a valid rock image.'}), 400
        
        # Get rock type and description
        rock_type = index[predicted_class]
        result = f'This is {rock_type} and it is {rock_info.get(rock_type, "a type of rock.")}'
        print('Prediction result:', result)

        return jsonify({'result': result, 'confidence': float(confidence)})  # Send result as JSON
    
    except Exception as e:
        print('Error:', e)
        return jsonify({'error': str(e)}), 500  # Send error as JSON

if __name__ == "__main__":
    app.run(debug=True)
