import streamlit as st

# Title
st.title("🎚️ Simple Slider Demo")

# Slider
value = st.slider("Select a value", min_value=0, max_value=100, value=50)

# Display selected value
st.write(f"You selected: {value}")
