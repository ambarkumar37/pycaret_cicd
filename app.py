# from flask import Flask,request, url_for, redirect, render_template, jsonify
# from pycaret.regression import *
# import pandas as pd
# import pickle
# import numpy as np
# import config

# app = Flask(__name__)

# model = load_model('deployment_28042020')
# cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

# @app.route('/')
# def home():
#     return render_template("home.html")

# @app.route('/predict',methods=['POST'])
# def predict():
#     int_features = [x for x in request.form.values()]
#     final = np.array(int_features)
#     data_unseen = pd.DataFrame([final], columns = cols)
#     prediction = predict_model(model, data=data_unseen, round = 0)
#     prediction = int(prediction.Label[0])
#     return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data = request.get_json(force=True)
#     data_unseen = pd.DataFrame([data])
#     prediction = predict_model(model, data=data_unseen)
#     output = prediction.Label[0]
#     return jsonify(output)

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)

from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import numpy as np
import config

app = Flask(__name__)

# Load trained PyCaret model
model = load_model('deployment_28042020')

# Base input columns expected from form/API
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

# Helper: add trig features if used during training
def add_trig_features(df):
    if 'age' in df and 'bmi' in df:
        df['age_sin'] = np.sin(df['age'].astype(float))
        df['age_cos'] = np.cos(df['age'].astype(float))
        df['bmi_sin'] = np.sin(df['bmi'].astype(float))
        df['bmi_cos'] = np.cos(df['bmi'].astype(float))
    return df

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns=cols)

    # Add trig features if needed
    data_unseen = add_trig_features(data_unseen)

    # Predict
    prediction = predict_model(model, data=data_unseen, round=0)
    prediction_value = int(prediction['prediction_label'][0])

    return render_template('home.html', pred=f'Expected Bill will be {prediction_value}')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    for num_col in ['age', 'bmi', 'children']:
        data[num_col] = float(data[num_col])
    data_unseen = pd.DataFrame([data])
    data_unseen = add_trig_features(data_unseen)
    prediction = predict_model(model, data=data_unseen)
    output = prediction['prediction_label'][0]
    return jsonify({'prediction': round(float(output), 2)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
