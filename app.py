from flask import Flask, render_template, request
from keras.models import load_model
import keras.utils as image
import numpy as np

app = Flask(__name__)

dic = {0 : 'Keyboard', 1 : 'Laptop', 2 : 'Mobile', 3 : 'Mouse', 4 : 'Television', 5 : 'Other'}

model = load_model('final-model-v2.h5')

model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224,224))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1, 224,224,3)
    y_pred = model.predict(i)
    p = np.argmax(y_pred,axis=1)
    return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return render_template("about.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict_page():
	return render_template("predict.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/uploads/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("predict.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)