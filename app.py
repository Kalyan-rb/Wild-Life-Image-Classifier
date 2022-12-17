# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 22:48:35 2022

@author: PavanB
"""

from flask import Flask, render_template,url_for,request,redirect
from werkzeug.utils import secure_filename
import os
import numpy as np
# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app= Flask(__name__,template_folder='Templates')

# Model saved with Keras model.save()
MODEL_PATH ='UpdatedWildLifeClassifier.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    dictt={'0':'Bald Eagle','1':'Black Bear','2':'Bobcat','3':'Canada Lynx','4':'Coulmbian Black tailed deer','5':'Cougar',
      '6':'Coyote','7':'Deer','8':'Elk','9':'Gray Fox','10':'Gray Wolf','11':'Mountain Beaver','12':'Nutria','13':'Racoon',
      '14':'Raven','15':'Red Fox','16':'Ring Tail','17':'Sea Lions','18':'Seals','19':'Virginia Opossum'}
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    test_input= np.expand_dims(x, axis=0)
   
    preds=model.predict(test_input)
    output=np.argmax(preds)
    output=str(output)
    preds=dictt.get(output)
 
    return preds



@app.route('/', methods=['GET','POST'])
def index():
    # Main page
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
      
        f.save(f'uploads/{f.filename}')
        
        file_path=f'uploads/{f.filename}'
        
        # Make prediction
        preds = model_predict(file_path, model)
        result= "Hey, Your looking at {}".format(preds)
        return render_template('index.html',result=result)
   
      



if __name__=='__main__':
    app.run(debug=True)