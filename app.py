from flask import Flask,render_template, request
from script import predictImage
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf

 # Define the custom loss deserialization function
def custom_loss_deserialization(config):
    if 'reduction' in config and config['reduction'] == 'auto':
        config['reduction'] = tf.keras.losses.Reduction.NONE  # Adjust as necessary
    return tf.keras.losses.deserialize(config)

    # Patch the deserialization method
tf.keras.losses.get = custom_loss_deserialization
app = Flask(__name__)

@app.route('/',methods =['GET'])
def hello():
    return render_template("index.html")

@app.route('/',methods =['POST'])
def predict():
    img = request.files['imageFile']
    imgPath = "images/"+ img.filename
    print(img.filename)
    img.save("./images/"+img.filename)
    # image = cv2.imread("/images/"+img.filename)
    classification = predictImage(imgPath)
    
    return render_template("index.html",prediction = classification)


if __name__ == "__main__" :
    app.run(port = 3000,debug=True)