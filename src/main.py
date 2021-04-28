#  Necessary Import---
from  flask import Flask,request,render_template,redirect,url_for,send_from_directory 
from werkzeug.utils import secure_filename
import os
import pandas as pandas
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import json
import psutil

#Import Ends---

app = Flask(__name__) # Initializaing App---  

model = load_model('model_covid.h5') # Loading the CNN Model---

# Initializing the image directory
UPLOAD_FOLDER = './images/'
ALLOWED_EXTENSIONS = set(['jpg','jpeg','png'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#End----

#Defining the Home route--
@app.route('/')
def home():
    return render_template('index.html')

#Ends--

#Defining the Upload X-ray Route---

@app.route('/uploader',methods = ['POST'])
def success():
    if request.method == 'POST': #check whether the incoming request is of POST type
        #Saving the incoming file to to image directory
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #Ends
        #Loading and Converting the image to appropriate format for prediction
        test_image = image.load_img('./images/'+filename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        #Ends
        result = model.predict(test_image) #Predicting the results
        res = ""
        x1 = 0
        y1= 0
        #Interpreting the predicting results
        if(int(result[0][0] == 0)):
            res = 'Negative'
            x1 = 0.8
            y1 = 0.2
        else:
            res = 'Postive'
            x1 = 0.2
            y1 = 0.8
        #print(type(result[0][0]))
        #Ends
        #Plotting the bargraph according to the predicted values
        x = ['Negative','Postive']
        y = [x1,y1]
        fig = go.Figure(data=[go.Bar(x=x, y=y)])
        fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',marker_line_width=1.5, opacity=0.6)
        fig.update_layout(title_text='Covid-19 Prediction Results')
        fig.write_image('static/'+filename)#saving the bargraph image into static directory
        #Ends
        return render_template('index.html', filename=filename,result=res,plot='static/'+filename)#Returing the results to index.html page

#Ends ----

#function to resutrn the filename of the uploaded image
@app.route('/images/<filename>')
def display_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
#ends--

#def predict(filename):
 #   print(filename)

# Main Function
if __name__ == '__main__':  
    app.run(debug = True)      
