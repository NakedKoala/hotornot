import logging
from flask import Flask
from flask import request
from flask import json
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import importlib
from model import Model
# import matplotlib.pyplot as plt
import tensorflow as tf    
import cv2     






app = Flask(__name__)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
model = Model()
graph = tf.get_default_graph()

def img_preparation(img):

    cv2.imwrite('tmp1.jpg',img)
    img = cv2.resize(img,(350,350))
    cv2.imwrite('tmp2.jpg',img)
    img = img / 255.0
    
    img_array = img.reshape(1,350,350,3)
    return img_array


@app.route('/', methods=['GET'])
def index():
    
    return "I am Alive"


@app.route('/predict', methods=['POST'])
def predict():
    bounding_offset = 15
    if request.method == 'POST':
        print("predict handler invoked")
        f = request.files['img_file']
        f.save('./query_img.jpg')

        img = cv2.imread('./query_img.jpg')
        

       

        # img = load_img("./query_img.jpg",target_size=(350, 350))

        # img = img_to_array(img)
        

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is ():
            pass 
        else:
            for (x,y,w,h) in faces:
                x_start = y - bounding_offset  if y - bounding_offset >= 0  else y
                x_end = y+h+bounding_offset  if y+h+bounding_offset <= img.shape[0]  else y+h
                y_start =  x - bounding_offset if x - bounding_offset >= 0 else x
                y_end = x+w+bounding_offset  if x+w+bounding_offset <= img.shape[1] else x+w
                
                img = img[x_start:x_end, y_start:y_end, :]    


        img_array = img_preparation(img)

        global graph
        with graph.as_default():
        
            score = model.predict(img_array)

            res = { "score": str(score[0][0]) }
            

        return json.dumps(res)





if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=8080)
