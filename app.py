# -*- coding: utf-8 -*-

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf


# Keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH ='/e3-smart-farmer/model_inception.h5'

# Load your trained model
model = load_model('app/model_inception.h5')




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Disease is Apple___Apple_scab"
    elif preds==1:
        preds="The Disease is Apple___Black_rot"
    elif preds==2:
        preds="The Disease is Apple___Cedar_apple_rust"
    elif preds==3:
        preds="The Disease is Apple___healthy"
    elif preds==4:
        preds="The Disease is Blueberry___healthy"
    elif preds==5:
        preds="The Disease is Cherry_(including_sour)___Powdery_mildew"
    elif preds==6:
        preds="The Disease is Cherry_(including_sour)___healthy"
    elif preds==7:
        preds="The Disease is Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
    elif preds==8:
        preds="The Disease is Corn_(maize)___Common_rust_"
    elif preds==9:
        preds="The Disease is Corn_(maize)___Northern_Leaf_Blight"
    elif preds==10:
        preds="The Disease is Corn_(maize)___healthy"
    elif preds==11:
        preds="The Disease is Grape___Black_rot"
    elif preds==12:
        preds="The Disease is Grape___Esca_(Black_Measles)"
    elif preds==13:
        preds="The Disease is Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"
    elif preds==14:
        preds="The Disease is Grape___healthy"
    elif preds==15:
        preds="The Disease is Orange___Haunglongbing_(Citrus_greening)"
    elif preds==16:
        preds="The Disease is Peach___Bacterial_spot"
    elif preds==17:
        preds="The Disease is Peach___healthy"
    elif preds==18:
        preds="The Disease is Pepper,_bell___Bacterial_spot"
    elif preds==19:
        preds="The Disease is Pepper,_bell___healthy"
    elif preds==20:
        preds="The Disease is Potato___Early_blight"
    elif preds==21:
        preds="The Disease is Potato___Late_blight"
    elif preds==22:
        preds="The Disease is Potato___healthy"
    elif preds==23:
        preds="The Disease is Raspberry___healthy"
    elif preds==24:
        preds="The Disease is Soybean___healthy"
    elif preds==25:
        preds="The Disease is Squash___Powdery_mildew"
    elif preds==26:
        preds="The Disease is Strawberry___Leaf_scorch"
    elif preds==27:
        preds="The Disease is Strawberry___healthy"
    elif preds==28:
        preds="The Disease is Tomato___Bacterial_spot" 
    elif preds==29:
        preds="The Disease is Tomato___Early_blight"
    elif preds==30:
        preds="The Disease is Tomato___Late_blight"
    elif preds==31:
        preds="The Disease is Tomato___Leaf_Mold"
    elif preds==32:
        preds="The Disease is Tomato___Septoria_leaf_spot"
    elif preds==33:
        preds="The Disease is Tomato___Spider_mites Two-spotted_spider_mite"
    elif preds==34:
        preds="The Disease is Tomato___Target_Spot"
    elif preds==35:
        preds="The Disease is Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    elif preds==36:
        preds="The Disease is Tomato___Tomato_mosaic_virus"
    elif preds==37:
        preds="The Disease is Tomato___healthy"
    
    
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

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5003,debug=True)
