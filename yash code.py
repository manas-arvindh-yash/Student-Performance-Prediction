import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="centered",
    page_icon="📊"
)

# Header
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>🎓 Student Performance Predictor</h1>
    <h4 style='text-align: center; color: gray;'>Upload your dataset to predict Total Score using Machine Learning</h4>
    <br>
""", unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Required columns
    required_columns = [
        'Gender', 'Age', 'Department', 'Attendance (%)',
        'Midterm_Score', 'Final_Score', 'Assignments_Avg',
        'Quizzes_Avg', 'Participation_Score', 'Projects_Score',
        'Study_Hours_per_Week', 'Extracurricular_Activities',
        'Internet_Access_at_Home', 'Parent_Education_Level',
        'Family_Income_Level', 'Stress_Level (1-10)',
        'Sleep_Hours_per_Night', 'Total_Score'
    ]

    if not all(col in df.columns for col in required_columns):
        st.error("❌ Your dataset is missing some required columns.")
        st.stop()

    df = df[required_columns]

    # Label Encoding
    cat_cols = [
        'Gender', 'Department', 'Extracurricular_Activities',
        'Internet_Access_at_Home', 'Parent_Education_Level',
        'Family_Income_Level'
    ]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Split
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

    # Layout
    st.markdown("### 📊 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("✅ RMSE", f"{rmse:.2f}")
    col2.metric("✅ R² Score", f"{r2:.2f}")

    # Predicted vs Actual Table
    st.markdown("### 🔍 Predicted vs Actual (Top 10)")
    results_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    }).reset_index(drop=True)
    st.dataframe(results_df.head(10).style.format({'Actual': '{:.1f}', 'Predicted': '{:.1f}'}))

    # Prediction Form
    st.markdown("### 🎯 Predict for a Custom Student")
    with st.form("predict_form"):
        custom_input = {}
        for col in X.columns:
            if col in cat_cols:
                options = list(encoders[col].classes_)
                selected = st.selectbox(f"{col}", options)
                custom_input[col] = encoders[col].transform([selected])[0]
            else:
                val = float(df[col].mean())
                custom_input[col] = st.number_input(f"{col}", value=val)

        submitted = st.form_submit_button("Predict Total Score")

        if submitted:
            input_df = pd.DataFrame([custom_input])
            pred_score = model.predict(input_df)[0]
            st.success(f"📚 Predicted Total Score: **{pred_score:.2f}**")

else:
    st.info("👆 Upload a CSV file to get started!")

# Footer
st.markdown("""
<hr style='border: 1px solid #ccc'>
<p style='text-align: center; color: #999;'>Built with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)
