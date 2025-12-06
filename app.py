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

# Try to load training target range for proper min-max rescaling (optional)
try:
    with open('target_range.json', 'r') as f:
        tr = json.load(f)
        t_min = float(tr.get('min'))
        t_max = float(tr.get('max'))
except Exception:
    t_min, t_max = None, None

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

    # Rescale/cap prediction into desired output range (3-8)
    out_min, out_max = 3.0, 8.0
    if t_min is not None and t_max is not None and t_max != t_min:
        # min-max rescale from training target range to out_min/out_max
        raw_rescaled = (raw_pred - t_min) / (t_max - t_min) * (out_max - out_min) + out_min
    else:
        # fallback: clamp into [out_min, out_max]
        raw_rescaled = max(out_min, min(out_max, float(raw_pred)))

    rounded = int(np.round(raw_rescaled))
    rounded = max(int(out_min), min(int(out_max), rounded))
    
    # Choose emoji based on quality
    # Emoji mapping for 3-8 scale
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
    else:  # 8
        emoji = '🤩'

    st.success(f"Predicted Quality: **{rounded}/8** {emoji}")
    st.info(f"Raw prediction (model): {raw_pred:.3f}")
    if t_min is None:
        st.warning('Training target range not found; result was clamped to 3–8. For smoother scaling, save training range as target_range.json from your notebook.')