from array import array

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
    symptom1 = request.form.get('symptom1')
    symptom2 = request.form.get('symptom2')
    symptom3 = request.form.get('symptom3')
    symptom4 = request.form.get('symptom4')
    symptom5 = request.form.get('symptom4')
    symptom6 = request.form.get('symptom4')
    symptom7 = request.form.get('symptom4')
    symptom8 = request.form.get('symptom4')
    symptom9 = request.form.get('symptom4')
    symptom10= request.form.get('symptom10')


    input_query = np.array(
        [symptom1, symptom2, symptom3, symptom4, symptom5, symptom6, symptom7, symptom8, symptom9, symptom10])

    array_2d = np.reshape(input_query,(1,-1))

    result = model.predict(array_2d)
    print(result)

    return jsonify({'prognosis': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
