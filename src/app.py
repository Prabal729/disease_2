import streamlit as st
import pandas as pd
from pathlib import Path

# Use shared theme and loaders
try:
    from .shared import set_page, inject_theme, load_artifacts, load_training_data
except ImportError:
    # Fallback when run as a script
    from shared import set_page, inject_theme, load_artifacts, load_training_data

set_page("🧬 Disease Predictor", "🧬")
inject_theme()

st.markdown("<h1 class='neon'>🧬 Disease Prediction</h1>", unsafe_allow_html=True)
st.caption("Toggle symptoms, predict likely diseases, and explore analytics.")

artifacts = load_artifacts()
model = artifacts["model"]
features = artifacts["features"]
label_enc = artifacts["label_encoder"]

X_train, y_train, X_valid, y_valid, X_all = load_training_data(features)

col1, col2, col3, col4 = st.columns(4)
with col1:
    total = int(X_all.shape[0]) if isinstance(X_all, pd.DataFrame) else 0
    st.markdown(f"<div class='glass'><h3>🧾 Records</h3><h2>{total:,}</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='glass'><h3>🧩 Features</h3><h2>{len(features)}</h2></div>", unsafe_allow_html=True)
with col3:
    num_classes = len(getattr(label_enc, 'classes_', []))
    st.markdown(f"<div class='glass'><h3>🦠 Diseases</h3><h2>{num_classes}</h2></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='glass'><h3>📦 Model</h3><h2>{'Loaded' if model is not None else 'Missing'}</h2></div>", unsafe_allow_html=True)

st.markdown("<h3 class='section'>Get Started</h3>", unsafe_allow_html=True)
st.markdown("- Open ‘Predictor’ page to toggle symptoms and get a prediction.\n- Explore ‘Data Explorer’ and ‘Analytics’ for insights.")

st.markdown("---")
st.caption("🧬 Enhanced UI • Glass + neon aesthetic • Multipage navigation in sidebar")