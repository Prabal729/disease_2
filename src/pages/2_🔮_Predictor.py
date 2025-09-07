import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import time
import json
try:
    from ..shared import set_page, inject_theme, load_artifacts, ui_toggle
except Exception:
    from shared import set_page, inject_theme, load_artifacts, ui_toggle

set_page("🔮 Predictor • Disease Predictor", "🧬")
inject_theme()

artifacts = load_artifacts()
model = artifacts["model"]
features = artifacts["features"]
label_enc = artifacts["label_encoder"]

st.markdown("<h1 class='neon'>🔮 Predictor</h1>", unsafe_allow_html=True)

if model is None or not features:
    st.error("Model or features are missing. Ensure artifacts exist in models/ and data/processed/ folders.")
else:
    st.markdown("<h3 class='section'>Symptoms</h3>", unsafe_allow_html=True)
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        query = st.text_input("Search symptoms", value="", placeholder="Type to filter symptoms...")
    with ctrl2:
        preset = st.selectbox("Preset", ["None", "Flu-like", "Cardio Risk", "Respiratory"])
    with ctrl3:
        apply_preset = st.button("Apply preset")

    q = (query or "").strip().lower()
    filtered = [f for f in features if q in f.lower()] if q else features

    # Toggle all filtered symptoms with a single button
    try:
        all_selected = all(bool(st.session_state.get(f"feat_{idx}", False)) for idx, feat in enumerate(features) if feat in filtered) if filtered else False
    except Exception:
        all_selected = False
    toggle_label = "Deselect All (filtered)" if all_selected else "Select All (filtered)"
    toggle_clicked = st.button(toggle_label, key="toggle_all_filtered")

    if apply_preset and preset != "None":
        tokens = {
            "Flu-like": ["fever", "cough", "fatigue", "headache", "throat", "aches"],
            "Cardio Risk": ["chest", "pressure", "hypertension", "heart", "palp", "bp"],
            "Respiratory": ["breath", "wheeze", "asthma", "oxygen", "spo2", "resp"],
        }[preset]
        for idx, feat in enumerate(features):
            key = f"feat_{idx}"
            if any(tok in feat.lower() for tok in tokens):
                st.session_state[key] = True

    if toggle_clicked:
        for idx, feat in enumerate(features):
            if feat in filtered:
                st.session_state[f"feat_{idx}"] = not all_selected

    st.caption(f"Showing {len(filtered)} of {len(features)} features")
    try:
        cols = st.columns(4, gap="small")
    except TypeError:
        cols = st.columns(4)
    input_data = {}
    for idx, feature in enumerate(features):
        if feature not in filtered:
            continue
        with cols[idx % 4]:
            st.markdown("<div class='feature'>", unsafe_allow_html=True)
            default_val = bool(st.session_state.get(f"feat_{idx}", False))
            checked = ui_toggle(feature, value=default_val, key=f"feat_{idx}")
            input_data[feature] = 1 if checked else 0
            st.markdown("</div>", unsafe_allow_html=True)

    selected_symptoms = [f for f, v in input_data.items() if v == 1]
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"<div class='glass'><h3>🧪 Selected Symptoms</h3><h2>{len(selected_symptoms)}</h2></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='glass'><h3>🧩 Total Features</h3><h2>{len(features)}</h2></div>", unsafe_allow_html=True)
    with right:
        try:
            est = (len(selected_symptoms) / max(1, len(features))) * 100.0
        except Exception:
            est = 0.0
        fig_live = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(est),
            title={'text': "Live Risk Estimate"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#a78bfa'},
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(148,163,184,0.20)'},
                    {'range': [33, 66], 'color': 'rgba(124,58,237,0.25)'},
                    {'range': [66, 100], 'color': 'rgba(34,211,238,0.30)'}
                ]
            }
        ))
        fig_live.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=280)
        st.plotly_chart(fig_live, width='stretch')

    predict_clicked = st.button("🚀 Run AI Prediction", type="primary")

    if predict_clicked:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i in range(0, 101, 5):
                time.sleep(0.01)
                progress_bar.progress(i)
                status_text.text("Running model..." if i > 40 else "Preparing input...")

            input_df = pd.DataFrame([input_data])
            input_df = input_df[[f for f in features if f in input_df.columns]]

            pred = model.predict(input_df)
            prob_pct = None
            try:
                p = model.predict_proba(input_df)[0]
                idx = int(np.argmax(p))
                prob_pct = float(p[idx]) * 100
                pred_encoded = int(pred[0]) if not hasattr(model, 'classes_') else model.classes_[idx]
            except Exception:
                pred_encoded = int(pred[0])

            try:
                disease = label_enc.inverse_transform([pred_encoded])[0] if label_enc is not None else str(pred_encoded)
            except Exception:
                disease = str(pred_encoded)

            progress_bar.empty(); status_text.empty()
            prob_text = f"{prob_pct:.2f}%" if prob_pct is not None else "N/A"
            bar_style = f"--target:{prob_pct:.2f}%" if prob_pct is not None else "--target:0%"
            st.markdown(
                f"""
                <div class='glass result-card'>
                  <div style='display:flex;align-items:center;justify-content:space-between;gap:0.75rem;'>
                    <div class='title'>Prediction Result</div>
                    <span class='pill'>Confidence {prob_text}</span>
                  </div>
                  <div style='font-size:1.6rem;font-weight:800;margin-top:0.25rem;letter-spacing:0.2px;'>{disease}</div>
                  <div style='margin-top:0.75rem;'>
                    <div class='confidence'><div class='meter' style='{bar_style}'></div></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Quick insights cards
            try:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"<div class='glass'><h3>🧪 Selected Symptoms</h3><h2>{len([f for f, v in input_data.items() if v == 1])}</h2></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='glass'><h3>🎯 Confidence</h3><h2>{prob_text}</h2></div>", unsafe_allow_html=True)
                with c3:
                    top_symptom = next((f for f, v in input_data.items() if v == 1), 'None')
                    st.markdown(f"<div class='glass'><h3>🔍 Top Symptom</h3><h2>{top_symptom}</h2></div>", unsafe_allow_html=True)
            except Exception:
                pass

            # Gauge + Recommendations layout
            left, right = st.columns([1, 1])

            with left:
                try:
                    gauge_val = float(prob_pct) if prob_pct is not None else 0.0
                except Exception:
                    gauge_val = 0.0
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=gauge_val,
                    title={'text': "Prediction Confidence"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': '#22d3ee'},
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(148,163,184,0.20)'},
                            {'range': [50, 80], 'color': 'rgba(124,58,237,0.25)'},
                            {'range': [80, 100], 'color': 'rgba(34,211,238,0.30)'}
                        ]
                    }
                ))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300)
                st.plotly_chart(fig, width='stretch')

            with right:
                try:
                    level = 'Low' if prob_pct is None or prob_pct < 50 else 'Moderate' if prob_pct < 80 else 'High'
                    recs = [
                        "Review symptom accuracy and completeness",
                        "Consider additional clinical tests if available",
                        "Monitor changes over the next 48 hours"
                    ]
                    if level == 'High':
                        recs.insert(0, "Consult a healthcare professional promptly")
                    elif level == 'Moderate':
                        recs.insert(0, "Schedule a follow-up consultation")
                    else:
                        recs.insert(0, "Maintain routine health monitoring")
                    st.markdown("<div class='glass'><h3 class='section'>💡 Recommendations</h3>" + "".join([f"<div>• {r}</div>" for r in recs]) + "</div>", unsafe_allow_html=True)
                except Exception:
                    pass

            try:
                out = Path(__file__).resolve().parents[1] / 'predictions.csv'
                selected_symptoms = [f for f, v in input_data.items() if v == 1]
                row = pd.DataFrame([
                    {
                        'timestamp': pd.Timestamp.utcnow().isoformat(),
                        'predicted_disease': disease,
                        'confidence_percent': prob_pct if prob_pct is not None else np.nan,
                        'num_symptoms': len(selected_symptoms),
                        'selected_symptoms': '; '.join(selected_symptoms),
                    }
                ])
                if out.exists():
                    prev = pd.read_csv(out)
                    pd.concat([prev, row], ignore_index=True).to_csv(out, index=False, encoding='utf-8')
                else:
                    row.to_csv(out, index=False, encoding='utf-8')
                st.success("✅ Prediction saved to predictions.csv")
            except Exception as e:
                st.warning(f"Could not save prediction: {e}")

            # Recent predictions summary
            try:
                out = Path(__file__).resolve().parents[1] / 'predictions.csv'
                if out.exists():
                    recent = pd.read_csv(out).tail(5)
                    st.markdown("<h3 class='section'>Recent Predictions</h3>", unsafe_allow_html=True)
                    st.dataframe(recent, width='stretch')
            except Exception:
                pass
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")