
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import base64
import streamlit as st
import base64



    df = pd.read_csv("cstperformance01.csv")
    return df

df = load_data()

# Columns required
required_columns = [
    'Gender', 'Age', 'Department', 'Attendance (%)',
    'Midterm_Score', 'Final_Score', 'Assignments_Avg',
    'Projects_Score', 'Study_Hours_per_Week',
    'Extracurricular_Activities','Quizzes_Avg',
    'Internet_Access_at_Home', 'Parent_Education_Level',
    'Family_Income_Level', 'Stress_Level (1-10)',
    'Sleep_Hours_per_Night', 'Total_Score'
]

if not all(col in df.columns for col in required_columns):
    st.error("❌ Dataset is missing required columns.")
    st.stop()

df = df[required_columns]

# Encode categorical columns
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

# Train/test split
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


