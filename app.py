from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # allow Wix (browser) to call this API

# Load saved model + encoders + columns
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

@app.route("/")
def home():
    return jsonify({"status":"API running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Build dataframe using same column order used in training
        if columns:
            row = []
            for col in columns:
                val = data.get(col)
                if val is None:
                    # numeric default fallback
                    row.append(0)
                else:
                    if col in encoders:
                        # encoder expects original label (string)
                        row.append(int(encoders[col].transform([val])[0]))
                    else:
                        row.append(val)
            X = pd.DataFrame([row], columns=columns)
        else:
            X = pd.DataFrame([data])
            for col, le in encoders.items():
                if col in X.columns:
                    X[col] = le.transform(X[col])

        pred = model.predict(X)[0]
        return jsonify({"prediction": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
