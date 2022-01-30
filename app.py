from array import array
from typing import Optional

import requests
from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    Symptom1 = request.form.get('Symptom1')
    Symptom2 = request.form.get('Symptom2')
    Symptom3 = request.form.get('Symptom3')
    Symptom4 = request.form.get('Symptom4')
    Symptom5 = request.form.get('Symptom5')
    Symptom6 = request.form.get('Symptom6')
    Symptom7 = request.form.get('Symptom7')
    Symptom8 = request.form.get('Symptom8')
    Symptom9 = request.form.get('Symptom9')
    Symptom10 = request.form.get('Symptom10')


    input_query = np.array(
        [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, Symptom6, Symptom7, Symptom8, Symptom9, Symptom10])

    array_2d = np.reshape(input_query,(1,-1))

    result = model.predict(array_2d)
    print(result)

    return jsonify({'Result': str(result)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
