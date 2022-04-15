from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('final.pkl', 'rb'))

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
    symptom5 = request.form.get('symptom5')
    symptom6 = request.form.get('symptom6')
    symptom7 = request.form.get('symptom7')

    input_query = np.array([[symptom1, symptom2, symptom3, symptom4, symptom5, symptom6, symptom7]])

    prognosis = model.predict(input_query)[0]

    return jsonify({'prognosis': str(prognosis)})


if __name__ == '__main__':
    app.run(debug=True)
