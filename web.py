


# Check model architecture
# -*- coding: utf-8 -*-


from __future__ import division , print_function
import os 
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions



import tensorflow as tf
global graph
graph = tf.compat.v1.get_default_graph()
from flask import Flask,request,render_template,redirect,url_for,jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
CORS(app)
MODEL_PATH = 'trained_model.h5'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Recreate the model architecture to match the saved weights
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

# Load the weights
model.load_weights('trained_model.h5')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

@app.route('/',methods = ["GET"] )
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
        if f:
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)

            img = image.load_img(file_path, target_size=(64, 64))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            prediction = model.predict(x)
            predicted_class = np.argmax(prediction, axis=1)[0]

            index = ['Basalt', 'Conglomerate', 'Dolostone', 'Gabbro', 'Gneiss', 'Granite',
                     'Limestone', 'Marble', 'Quartzite', 'Rhyolite', 'Sandstone', 'Shale', 'Slate']

            rock_type = index[predicted_class]

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

            result = f'This is {rock_type} and it is {rock_info[rock_type]}'
            print('Prediction result:', result)
            return jsonify({'result': result})  # Send result as JSON
        else:
            return jsonify({'error': 'File not found'}), 400

    except Exception as e:
        print('Error:', e)
        return jsonify({'error': str(e)}), 500  # Send error as JSON


if __name__ == "__main__":
    app.run(debug=True)

