import numpy as np
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, request, jsonify, render_template
import pickle
import cv2 as cv
import tensorflow as tf
from werkzeug.utils import secure_filename

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

global graph
graph = tf.get_default_graph()
# classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    image = request.files['image']
    filename = secure_filename(image.filename)

    image.stream.seek(0) # seek to the beginning of file
    image_byte = image.read()
    npimg = np.fromstring(image_byte, np.uint8)
    file = cv.imdecode(npimg, cv.IMREAD_COLOR)
    # cv.imshow('', file)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    file = cv.resize(file, (IMG_SIZE, IMG_SIZE))
    file = np.array([file]) / 255.0
    # file = np.array([file])

    with graph.as_default():
        mobilenet = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, weights="imagenet")
        mobilenet.trainable = False
        prediction = mobilenet.predict(file, batch_size=file.shape[0])

    top_5_classes_index = np.argsort(prediction)[0 , ::-1][:5]+1

    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    labels = np.array(open(labels_path).read().splitlines())
    top_5_classes = labels[top_5_classes_index]
    class_name = ", ".join(top_5_classes)
    # return render_template('index.html', prediction_text='The image is a {}'.format(class_name))
    return render_template('index.html', prediction_text='The image can be: '+class_name)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)