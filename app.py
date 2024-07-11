from flask import Flask,render_template, request
from script import predictImage
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf

delete = False
 # Define the custom loss deserialization function
def custom_loss_deserialization(config):
    if 'reduction' in config and config['reduction'] == 'auto':
        config['reduction'] = tf.keras.losses.Reduction.NONE  # Adjust as necessary
    return tf.keras.losses.deserialize(config)

    # Patch the deserialization method
tf.keras.losses.get = custom_loss_deserialization
app = Flask(__name__, static_url_path="/static")
# Ensure the static folder exists
file = os.path.join( 'Image')
print(file)
if not os.path.exists(file):
    os.makedirs(file)

@app.route('/', methods=['GET'])
def hello():
    imgp = os.path.join( 'Image/masks-happy-sad.jpg')
    return render_template("index.html", imgp=imgp)

@app.route('/',methods =['POST'])
def predict():
    img = request.files['imageFile']
    imgPath = ( file + '/'+ img.filename)
    print(imgPath)
    img.save("./static/" + imgPath)
    # image = cv2.imread("/Image/"+img.filename)
    classification = predictImage(imgPath)
    # imgr = imgPath
    delete = True
    return render_template("index.html",prediction = classification,imgp = imgPath)


@app.after_request
def after_request_callback(response):
    image_folder = "static/Image"
    global delete
    if delete:
        specific_file = os.path.join(image_folder, 'masks-happy-sad.jpg')
        for file_name in os.listdir(image_folder):
            file_n = os.path.join(image_folder, file_name)
            if os.path.isfile(file_n) and file_n != specific_file:
                print('Deleting file:', file_n)
                os.remove(file_n)
        # Reset the flag after deleting files
        delete = False
    return response


# if __name__ == "__main__" :
#     app.run(port = 3000,debug=True)