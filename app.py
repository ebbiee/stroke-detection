"""
Then open: http://127.0.0.1:5000
"""

from flask import Flask, request, render_template, jsonify
import pickle, json, os
import numpy as np
import pandas as pd
import shap

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load artifacts saved by the training script ───────────────────────────────
MODEL_PATH        = os.path.join('model', 'best_model.pkl')
PREPROCESSOR_PATH = os.path.join('model', 'preprocessor.pkl')
FEATURES_PATH     = os.path.join('model', 'feature_names.json')
NAME_PATH         = os.path.join('model', 'best_model_name.json')

with open(MODEL_PATH,        'rb') as f: model        = pickle.load(f)
with open(PREPROCESSOR_PATH, 'rb') as f: preprocessor = pickle.load(f)
with open(FEATURES_PATH,     'r')  as f: feature_names = json.load(f)
with open(NAME_PATH,         'r')  as f: best_model_name = json.load(f)['name']

print(f"✅ Model loaded        : {best_model_name}")
print(f"✅ Preprocessor loaded : OHE + StandardScaler")
print(f"✅ Features            : {len(feature_names)}")

# Column definitions — must match training script exactly
CATEGORICAL_COLS = ['gender', 'ever_married', 'work_type',
                    'Residence_type', 'smoking_status']
NUMERICAL_COLS   = ['age', 'hypertension', 'heart_disease',
                    'avg_glucose_level', 'bmi']


def form_to_dataframe(form_data):
    row = {
        'gender':            form_data.get('gender', 'Male'),
        'age':               float(form_data.get('age', 0)),
        'hypertension':      int(form_data.get('hypertension', 0)),
        'heart_disease':     int(form_data.get('heart_disease', 0)),
        'ever_married':      form_data.get('ever_married', 'No'),
        'work_type':         form_data.get('work_type', 'Private'),
        'Residence_type':    form_data.get('Residence_type', 'Urban'),
        'avg_glucose_level': float(form_data.get('avg_glucose_level', 100)),
        'bmi':               float(form_data.get('bmi', 25)),
        'smoking_status':    form_data.get('smoking_status', 'never smoked'),
    }
    return pd.DataFrame([row])


def preprocess_and_predict(raw_df):
    X_processed  = preprocessor.transform(raw_df)       # scale + encode
    prediction   = int(model.predict(X_processed)[0])
    probability  = float(model.predict_proba(X_processed)[0][1]) * 100
    return prediction, round(probability, 2)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_df              = form_to_dataframe(request.form)
        prediction, prob    = preprocess_and_predict(raw_df)

        result = {
            'prediction' : prediction,
            'label'      : 'HIGH RISK — Stroke Detected' if prediction == 1
                           else 'LOW RISK — No Stroke Detected',
            'probability': prob,
            'color'      : '#e74c3c' if prediction == 1 else '#27ae60',
            'icon'       : '⚠️' if prediction == 1 else '✅',
        }
        return render_template('result.html', result=result, form=request.form)

    except Exception as e:
        result = {'label': f'Error: {str(e)}', 'color': '#e67e22',
                  'icon': '❌', 'probability': 0, 'prediction': -1}
        return render_template('result.html', result=result, form=request.form)



@app.route('/api/predict', methods=['POST'])
def api_predict():
    data   = request.get_json(force=True)
    raw_df = form_to_dataframe(data)
    prediction, prob = preprocess_and_predict(raw_df)

    return jsonify({
        'stroke_prediction' : prediction,
        'stroke_probability': round(prob / 100, 4),
        'risk_label'        : 'High Risk' if prediction == 1 else 'Low Risk',
        'model_used'        : best_model_name
    })



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
