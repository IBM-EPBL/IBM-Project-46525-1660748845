from flask import Flask, request, render_template, flash
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf


app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER']= 'uploads/'
model = load_model("./models/mnistCNN.h5")
@app.route('/')
def batch():
    return render_template("index.html")

@app.route('/web')
def batch2():
    return render_template("web.html")

@app.route('/web',methods=['GET','POST'])
def web():
    imagefile = request.files['imagefile']
    image_path ="./uploads/"+imagefile.filename
    imagefile.save(image_path)
    img = image.load_img(image_path).convert("L")
    img = img.resize((28, 28))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 28, 28, 1)
    y_pred = model.predict(im2arr)
    pred = np.argmax(y_pred, axis=1)
    index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    output = str(index[pred[0]])
    return render_template('web.html', prediction=output)





if __name__=="__main__":
    app.run(debug=True)