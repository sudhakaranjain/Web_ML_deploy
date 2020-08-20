import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, request, jsonify, render_template
import pickle
# import urllib.request
import cv2 as cv
import tensorflow as tf
from werkzeug.utils import secure_filename

global graph
graph = tf.get_default_graph()
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))
UPLOAD_FOLDER = 'static/uploads/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # int_features = [int(x) for x in request.form.values()]
    image = request.files['image']
    filename = secure_filename(image.filename)
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    image.stream.seek(0) # seek to the beginning of file
    image_byte = image.read()
    npimg = np.fromstring(image_byte, np.uint8)
    file = cv.imdecode(npimg, cv.IMREAD_COLOR)
    # cv.imshow('', file)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    file = cv.resize(file, (32,32))
    file = np.array([file]) / 255.0
    # print(file.shape)
    # keras.backend.clear_session()
    with graph.as_default():
        model = tf.keras.models.load_model('./model.h5')
        prediction = model.predict(file, batch_size=file.shape[0])
    output = np.argmax(prediction, axis=1)
    class_name = classes[output[0]]
    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(class_name))

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