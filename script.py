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

def predictImage(imgPath):
    # Load the model without compiling
    model_path = os.path.join('models', 'happysadimagemodel2.h5')
    new_model = load_model(model_path, compile=False)

    # Recompile the model with a valid loss function and other necessary parameters
    new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Read and preprocess the image
    img = cv2.imread(imgPath)
    if img is None:
        raise ValueError(f"Image not loaded correctly: {imgPath}"+ " \n\n\n\n\n "+ imgPath)
    
    resizeimg = cv2.resize(img, (256, 256))
    yhat = new_model.predict(np.expand_dims(resizeimg / 255.0, 0))

    if yhat > 0.5:
        mood = "sad"
    else:
        mood = "happy"

    return mood

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Failed to capture image")
    #         break
        
    #     if currframe % 30 == 0:  # Process every 30th frame
    #         resize = cv2.resize(frame, (256, 256))
    #         yhat = new_model.predict(np.expand_dims(resize / 255.0, 0))
    #         print(yhat)
            
    #         if yhat > 0.5:
    #             mood = "sad"
    #         else:
    #             mood = "happy"
    #         print(f"You are {mood}")

    #     cv2.putText(
    #         frame,  # numpy array on which text is written
    #         mood,  # text
    #         (10, 50),  # position at which writing has to start
    #         cv2.FONT_HERSHEY_SIMPLEX,  # font family
    #         3,  # font size
    #         (209, 80, 0, 255),  # font color
    #         3  # font stroke
    #     )
        
    #     currframe += 1
    #     cv2.imshow('frame', frame)
        
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

