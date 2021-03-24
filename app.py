from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)



from tensorflow.keras.models import load_model
import tensorflow as tf

model = tf.keras.models.load_model('modelCNN.h5')
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
print('Model loaded. Start serving...')
'''
# You can also use pretrained model from Keras
# Check https://keras.io/applications/
from keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet')
model.save('modelkeras.h5')
'''
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path)
    img= img.resize((200,200))
    # Preprocessing the image
    #x=img.resize((200,200))  #added
    #x = image.img_to_array(img)
    #x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')
    img=np.expand_dims(img,axis=0)
    preds = model.predict(img)
    if preds[0][0] == 0.0:
        return "No violence"
    else:
        return "Violence"
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(basepath)
        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)
