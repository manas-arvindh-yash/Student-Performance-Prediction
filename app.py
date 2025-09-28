from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)

# Load model (works if you saved either a model object or a dict with keys: model, encoders, columns)
with open("student_model.pkl", "rb") as f:
    saved = pickle.load(f)

if isinstance(saved, dict):
    model = saved.get("model")
    encoders = saved.get("encoders", {})
    columns = saved.get("columns", None)
else:
    model = saved
    encoders = {}
    columns = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "API running",
        "note": "Use POST /predict with JSON body. Example keys: Midterm_Score, Final_Score, Assignments_Avg, Quizzes_Avg, Projects_Score, Attendance, Study_Hours_per_Week, Sleep_Hours_per_Night, Stress_Level"
    })

def build_dataframe_from_input(data):
    # data is a dict (from JSON)
    if columns:
        row = []
        for col in columns:
            val = data.get(col)
            if val is None:
                # default numeric fallback
                row.append(0)
            else:
                if col in encoders:
                    # attempt to transform label (if encoder exists)
                    try:
                        transformed = encoders[col].transform([val])[0]
                        row.append(int(transformed))
                    except Exception:
                        # if transform fails, try numeric conversion
                        try:
                            row.append(float(val))
                        except Exception:
                            row.append(val)
                else:
                    try:
                        row.append(float(val))
                    except Exception:
                        row.append(val)
        return pd.DataFrame([row], columns=columns)
    else:
        df = pd.DataFrame([data])
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])
        return df

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        X = build_dataframe_from_input(data)
        pred = model.predict(X)[0]
        return jsonify({"prediction": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

