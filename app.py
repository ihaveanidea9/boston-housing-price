import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
model = pickle.load(open('boston-housing-model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))


messages = [{'title': 'Message One',
             'content': 'Message One Content'},
            {'title': 'Message Two',
             'content': 'Message Two Content'}
            ]

@app.route('/')
def index():
    return render_template('index.html', messages=messages)

@app.route('/about')
def about():
    return render_template('about.html', messages='About Page')


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    raw_data= np.array(data).reshape(1,-1)
    print(raw_data)
    final_input = scaler.transform(raw_data)
    output = model.predict(final_input)[0]
    print(output)
    return render_template('index.html', prediction_text='Predicted House price iss {}'.format(output))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    raw_data= np.array(list(data.values())).reshape(1,-1)
    print(raw_data)
    input_data = scaler.transform(raw_data)
    output = model.predict(input_data)
    print(output)
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)