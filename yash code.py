import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit Page Setup
st.set_page_config(page_title="📊 Student Score Predictor", layout="centered")
st.title("🎓 Student Total Score Predictor (Regression)")

# File Upload
uploaded_file = st.file_uploader("📁 Upload your student dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Required Columns
    required_columns = [
        'Gender', 'Age', 'Department', 'Attendance (%)',
        'Midterm_Score', 'Final_Score', 'Assignments_Avg',
        'Quizzes_Avg', 'Participation_Score', 'Projects_Score',
        'Study_Hours_per_Week', 'Extracurricular_Activities',
        'Internet_Access_at_Home', 'Parent_Education_Level',
        'Family_Income_Level', 'Stress_Level (1-10)',
        'Sleep_Hours_per_Night', 'Total_Score'
    ]

    # Validate Columns
    if not all(col in df.columns for col in required_columns):
        st.error("❌ Uploaded file is missing required columns.")
        st.stop()

    df = df[required_columns]

    # Label Encoding for Categorical Columns
    categorical_cols = [
        'Gender', 'Department', 'Extracurricular_Activities',
        'Internet_Access_at_Home', 'Parent_Education_Level',
        'Family_Income_Level'
    ]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Split Data
    X = df.drop('Total_Score', axis=1)
    y = df['Total_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Performance")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("R² Score", f"{r2:.2f}")

    # Show Actual vs Predicted
    st.subheader("🧾 Predictions vs Actual")
    results_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    }).reset_index(drop=True)
    st.dataframe(results_df.head(10))

    # Custom Input Prediction
    st.subheader("🎯 Predict for a Custom Student")
    custom_input = {}
    for col in X.columns:
        if col in categorical_cols:
            options = list(encoders[col].classes_)
            selected = st.selectbox(f"{col}", options)
            encoded_value = encoders[col].transform([selected])[0]
            custom_input[col] = encoded_value
        else:
            default_val = float(df[col].mean())
            custom_input[col] = st.number_input(f"{col}", value=default_val)

    if st.button("Predict Total Score"):
        input_df = pd.DataFrame([custom_input])
        prediction = model.predict(input_df)[0]
        st.success(f"📚 Predicted Total Score: {prediction:.2f}")

else:
    st.info("👆 Please upload a CSV file to continue.")
