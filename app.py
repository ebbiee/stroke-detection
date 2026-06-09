"""
page url: http://127.0.0.1:5000
"""

from flask import Flask, request, render_template, jsonify
import pickle, json, os
import numpy as np
import pandas as pd
import shap


app = Flask(__name__)

MODEL_PATH        = os.path.join('model', 'best_model.pkl')
PREPROCESSOR_PATH = os.path.join('model', 'preprocessor.pkl')
FEATURES_PATH     = os.path.join('model', 'feature_names.json')
NAME_PATH         = os.path.join('model', 'best_model_name.json')
THRESHOLD_PATH    = os.path.join('model', 'best_threshold.json')
X_TRAIN_PATH      = os.path.join('model', 'X_train_sample.csv')

with open(MODEL_PATH,        'rb') as f: model           = pickle.load(f)
with open(PREPROCESSOR_PATH, 'rb') as f: preprocessor    = pickle.load(f)
with open(FEATURES_PATH,     'r')  as f: feature_names   = json.load(f)
with open(NAME_PATH,         'r')  as f: best_model_name = json.load(f)['name']
with open(THRESHOLD_PATH,    'r')  as f: best_threshold  = json.load(f)['threshold']

if best_model_name in ("Random Forest", "XGBoost"):
    explainer = shap.TreeExplainer(model)
else:
    X_train_background = pd.read_csv(X_TRAIN_PATH)
    explainer = shap.LinearExplainer(model, masker=X_train_background,
                                     feature_perturbation="interventional")

# Column definitions — must match the notebook exactly
CATEGORICAL_COLS = ['gender', 'ever_married', 'work_type',
                    'Residence_type', 'smoking_status']
NUMERICAL_COLS   = ['age', 'hypertension', 'heart_disease',
                    'avg_glucose_level', 'bmi']

# ── Map every encoded feature name → a human-readable group label ──────────
# These are the ONLY features your preprocessor actually produces:
#   OHE (drop='first') over CATEGORICAL_COLS  +  StandardScaler over NUMERICAL_COLS
FEATURE_GROUP_MAP = {
    # gender (Male dropped as reference, Female kept)
    'gender_Male'                    : 'Gender',
    'gender_Other'                   : 'Gender',
    'gender_Female'                  : 'Gender',

    # ever_married (No dropped as reference)
    'ever_married_Yes'               : 'Marital Status',

    # work_type (Govt_job dropped as reference)
    'work_type_Never_worked'         : 'Work Type',
    'work_type_Private'              : 'Work Type',
    'work_type_Self-employed'        : 'Work Type',
    'work_type_children'             : 'Work Type',

    # Residence_type (Rural dropped as reference)
    'Residence_type_Urban'           : 'Residence Type',

    # smoking_status (Unknown dropped as reference)
    'smoking_status_formerly smoked' : 'Smoking Status',
    'smoking_status_never smoked'    : 'Smoking Status',
    'smoking_status_smokes'          : 'Smoking Status',

    # Numerical — kept as-is (scaled but still individual features)
    'age'                            : 'Age',
    'hypertension'                   : 'Hypertension',
    'heart_disease'                  : 'Heart Disease',
    'avg_glucose_level'              : 'Average Glucose Level',
    'bmi'                            : 'BMI',
}

# Human-readable labels shown to the user for each input field
FEATURE_DISPLAY_NAMES = {
    'Gender'               : 'Gender',
    'Marital Status'       : 'Marital Status',
    'Work Type'            : 'Work Type',
    'Residence Type'       : 'Residence Type',
    'Smoking Status'       : 'Smoking Status',
    'Age'                  : 'Age',
    'Hypertension'         : 'Hypertension',
    'Heart Disease'        : 'Heart Disease',
    'Average Glucose Level': 'Average Glucose Level',
    'BMI'                  : 'BMI',
}


def form_to_dataframe(form_data):
    """Collect raw form/JSON values into a single-row DataFrame."""
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


def get_shap_factors(X_processed_df):
    """
    Compute per-input SHAP contributions and return them ranked from
    most risk-increasing (#1) to most risk-decreasing (#8).

    - SHAP values for all encoded dummy columns belonging to the same
      original input field are SUMMED into one grouped value.
    - Positive total  → this input pushes risk UP   (increased)
    - Negative total  → this input pulls risk DOWN  (decreased)
    - NO filtering threshold — every input the user provided is shown.
    - Sort: descending by raw SHAP (most increased → most decreased).
    """
    shap_values = explainer.shap_values(X_processed_df)

    # TreeExplainer on binary classifier may return [neg_class_arr, pos_class_arr]
    if isinstance(shap_values, list):
        sv = shap_values[1][0]   # positive-class SHAP for the single row
    else:
        sv = shap_values[0]

    # Accumulate SHAP values per human-readable group
    grouped = {}
    for feat, val in zip(feature_names, sv):
        group_name = FEATURE_GROUP_MAP.get(feat, feat)   # fallback = raw name
        grouped[group_name] = grouped.get(group_name, 0.0) + float(val)

    # Build factor list — NO threshold filter so Age (and every other input) always appears
    factors = [
        {
            'feature'   : group_name,
            'direction' : 'increased' if total_shap > 0 else 'decreased',
            'shap_value': round(total_shap, 4),
            'abs_value' : abs(total_shap),
        }
        for group_name, total_shap in grouped.items()
    ]

    # Sort descending by raw SHAP value:
    #   rank 1  = biggest risk increaser
    #   rank 8  = biggest risk decreaser
    factors.sort(key=lambda x: -x['shap_value'])

    return factors[:8]


@app.route('/api/shap_debug', methods=['POST'])
def shap_debug():
    """
    Debug endpoint — returns the raw grouped SHAP values for every feature
    so you can verify nothing is being dropped or miscalculated.
    Hit with: POST /api/shap_debug  (same JSON body as /api/predict)
    """
    data   = request.get_json(force=True)
    raw_df = form_to_dataframe(data)

    X_processed    = preprocessor.transform(raw_df)
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    shap_values = explainer.shap_values(X_processed_df)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    raw = {feat: round(float(val), 6) for feat, val in zip(feature_names, sv)}

    grouped = {}
    for feat, val in zip(feature_names, sv):
        group_name = FEATURE_GROUP_MAP.get(feat, feat)
        grouped[group_name] = grouped.get(group_name, 0.0) + float(val)
    grouped_rounded = {k: round(v, 6) for k, v in grouped.items()}

    return jsonify({
        'feature_names_loaded' : feature_names,
        'raw_shap_per_column'  : raw,
        'grouped_shap'         : grouped_rounded,
        'shap_base_value'      : round(float(
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        ), 6),
    })


def preprocess_and_predict(raw_df):
    X_processed    = preprocessor.transform(raw_df)
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    proba        = float(model.predict_proba(X_processed_df)[0][1])
    prediction   = int(proba >= best_threshold)
    probability  = round(proba * 100, 2)
    shap_factors = get_shap_factors(X_processed_df)

    return prediction, probability, shap_factors


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_df                         = form_to_dataframe(request.form)
        prediction, prob, shap_factors = preprocess_and_predict(raw_df)

        result = {
            'prediction'  : prediction,
            'label'       : 'HIGH RISK — Stroke Detected' if prediction == 1
                            else 'LOW RISK — No Stroke Detected',
            'probability' : prob,
            'color'       : '#e74c3c' if prediction == 1 else '#27ae60',
            'shap_factors': shap_factors,
        }
        return render_template('result.html', result=result, form=request.form)

    except Exception as e:
        result = {
            'label'      : f'Error: {str(e)}',
            'color'      : '#e67e22',
            'probability': 0,
            'prediction' : -1,
        }
        return render_template('result.html', result=result, form=request.form, shap_factors=[])


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data                           = request.get_json(force=True)
    raw_df                         = form_to_dataframe(data)
    prediction, prob, shap_factors = preprocess_and_predict(raw_df)

    return jsonify({
        'stroke_detection'  : prediction,
        'stroke_probability': round(prob / 100, 4),
        'risk_label'        : 'High Risk' if prediction == 1 else 'Low Risk',
        'model_used'        : best_model_name,
        'top_factors'       : [
            {
                'rank'      : i + 1,
                'feature'   : f['feature'],
                'direction' : f['direction'],
                'shap_value': f['shap_value'],
            }
            for i, f in enumerate(shap_factors)
        ],
    })


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)