import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np

# Načti model, scaler a seznam features v správném pořadí
mlp = joblib.load('wine_model.pkl')
deeper_scaler = joblib.load('wine_scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

with open('feature_ranges.json', 'r') as f:
    ranges = json.load(f)

st.title("🍷 Wine Quality Predictor")
st.write("Posunuj slidery a predikuj kvalitu vína")

# Slidery pro všechny features v správném pořadí
values = {}
for feat in feature_names:
    if feat in ranges:
        lo, hi, step = ranges[feat]
        values[feat] = st.slider(feat, lo, hi, (lo + hi) / 2, step)

# Predikce
if st.button("Predict Quality"):
    # Sestav DataFrame v správném pořadí features
    df = pd.DataFrame([values])[feature_names]
    df_scaled = deeper_scaler.transform(df)
    raw_pred = mlp.predict(df_scaled)[0]
    rounded = int(np.round(raw_pred))
    rounded = max(0, min(10, rounded))  # clamp 0-10
    
    # Choose emoji based on quality
    if rounded <= 3:
        emoji = '😢'
    elif rounded == 4:
        emoji = '😕'
    elif rounded == 5:
        emoji = '😐'
    elif rounded == 6:
        emoji = '🙂'
    elif rounded == 7:
        emoji = '😃'
    else:  # 8-10
        emoji = '🤩'
    
    st.success(f"Predicted Quality: **{rounded}/10** {emoji}")
    st.info(f"Raw prediction: {raw_pred:.3f}")