from tensorflow import keras 
from flask import Flask
from flask import render_template
from flask import request
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf

from tensorflow import keras 
model = keras.models.load_model('./models/Vgg16-023-0.0384-0.9943.hdf5')

def preprocessing_image(img):
    resize_img = cv2.resize(img, (512,512))
    img_array_expanded_dims = np.expand_dims(resize_img, axis=0)
    return tf.keras.applications.vgg19.preprocess_input(img_array_expanded_dims)

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('image_upload.html')

@app.route('/LandMark', methods = ['GET', 'POST'])
def Cls_LandMark():
    if request.method == 'POST':
        img = request.files['chooseFile']

        print(type(img))
        
        p_img = preprocessing_image()
        prediction = model.predict(p_img)[0]
        pred_cls = prediction.argmax(axis=-1)

        return render_template('Search.html')
    
    else:
        return render_template('image_upload.html')

app.run(host='0.0.0.0', port=5031)