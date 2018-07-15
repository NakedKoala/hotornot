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
import tensorflow as tf 




app = Flask(__name__)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

model = Model()
graph = tf.get_default_graph()

def img_preparation(img):
    
    
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1,350,350,3)
    return img_array


@app.route('/', methods=['GET'])
def index():
    
    return "I am Alive"


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print("predict handler invoked")
        f = request.files['img_file']
        f.save('./query_img.jpg')

        img = load_img("./query_img.jpg",target_size=(350, 350))
        img_array = img_preparation(img)

        global graph
        with graph.as_default():
        
            score = model.predict(img_array)

            res = { "score": str(score[0][0]) }
            

        return json.dumps(res)





if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=8080)
