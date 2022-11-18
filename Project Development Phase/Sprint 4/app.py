from flask import Flask, request, render_template, flash
from PIL import ImageOps
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__, template_folder='template')
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
    img = ImageOps.grayscale(img)
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    im2arr = np.array(img)
    im2arr= im2arr/255.0
    im2arr = im2arr.reshape(1, 28, 28, 1)
    result = model.predict(im2arr)
    best = np.argmax(result, axis=1)[0]
    pred = list(map(lambda x: round(x * 100, 2), result[0]))
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    pred_val = list(zip(values,pred))
    best = pred_val.pop(best)
    return render_template('web.html',prediction=best)


if __name__=="__main__":
    app.run(debug=True)