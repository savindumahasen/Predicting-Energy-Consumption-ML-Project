import pickle
from flask import Flask,request,jsonify,render_template,app, url_for
import pandas as pd
import numpy as np


app=Flask(__name__)

## Load the model
regression_model=pickle.load(open('regression.pkl','rb'))
scaling=pickle.load(open('scaler.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    data=request.json['data']
    data = np.array(list(data.values())).reshape(1,-1)
    print(data)
    new_data=scaling.transform(data)
    predictions=regression_model.predict(new_data)
    print(predictions[0])
    return jsonify(predictions[0])


if __name__=="__main__":
    app.run(debug=True, port=5000)

