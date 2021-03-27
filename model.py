#backend
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__) #initializing

model=pickle.load(open('model.pkl','rb')) #loading model


@app.route('/') #url path
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET']) #get post method 
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)
    output=round(prediction[0],2)
    return render_template('index.html',pred='Insurance Cost will be Rs {}/- only'.format(output))
   


if __name__ == '__main__':
    app.run(debug=True)