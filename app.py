from tensorflow import keras 
from flask import Flask
from flask import render_template
from flask import request
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
import os

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
        img = request.files['chooseFile'].read()
        
        img_bytes = np.fromstring(img, dtype = np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_UNCHANGED)
                        
        p_img = preprocessing_image(img)
        prediction = model.predict(p_img)[0]
        pred_cls = prediction.argmax(axis=-1)

        if pred_cls == 0:
            result = '63빌딩'
        elif pred_cls == 1:
            result = '남산타워'
        elif pred_cls == 2:
            result = '경복궁'
        elif pred_cls == 3:
            result = '광장시장'
        elif pred_cls == 4:
            result = 'four'
        elif pred_cls == 5:
            result = '뚝섬한강공원'
        elif pred_cls == 6:
            result = '롯데타워'
        elif pred_cls == 7:
            result = '봉은사'
        elif pred_cls == 8:
            result = '북촌한옥마을'
        else:
            result = '서울숲'

        return render_template('Search.html', result=result)
    
    else:
        return render_template('image_upload.html')

app.run(host='0.0.0.0', port=5031)