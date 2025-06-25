import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="📊 Student Score Predictor", layout="centered")
st.title("🎓 Student Total Score Predictor")

# Load CSV
@st.cache_data
def load_data():
    df = pd.read_csv("/mnt/data/cstperformance01.csv")
    return df

df = load_data()

# Select needed columns
c_needed = [
    'Gender', 'Age', 'Department', 'Attendance (%)',
    'Midterm_Score', 'Final_Score', 'Assignments_Avg',
    'Quizzes_Avg', 'Participation_Score', 'Projects_Score',
    'Study_Hours_per_Week', 'Extracurricular_Activities',
    'Internet_Access_at_Home', 'Parent_Education_Level',
    'Family_Income_Level', 'Stress_Level (1-10)',
    'Sleep_Hours_per_Night', 'Total_Score'
]

if not all(col in df.columns for col in c_needed):
    st.error("❌ Some required columns are missing in the dataset.")
    st.stop()

df = df[c_needed]

# Encode categorical columns
c_cols = [
    'Gender', 'Department', 'Extracurricular_Activities',
    'Internet_Access_at_Home', 'Parent_Education_Level',
    'Family_Income_Level'
]
encoders = {}
for col in c_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split data
X = df.drop('Total_Score', axis=1)
y = df['Total_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.metric("RMSE", f"{rmse:.2f}")
st.metric("R² Score", f"{r2:.2f}")

# Show predictions
st.subheader("🧾 Predictions vs Actual")
results_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
}).reset_index(drop=True)
st.dataframe(results_df.head(10))

# Predict custom input
st.subheader("🎯 Predict for a Custom Student")
custom_input = {}
for col in X.columns:
    if col in c_cols:
        options = list(encoders[col].classes_)
        selected = st.selectbox(f"{col}", options)
        custom_input[col] = encoders[col].transform([selected])[0]
    else:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        custom_input[col] = val

if st.button("Predict Total Score"):
    input_df = pd.DataFrame([custom_input])
    prediction = model.predict(input_df)[0]
    st.success(f"📚 Predicted Total Score: {prediction:.2f}")
