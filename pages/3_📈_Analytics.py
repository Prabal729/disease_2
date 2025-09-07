import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
try:
    from src.shared import set_page, inject_theme, load_artifacts, load_training_data, decode_labels
except Exception:
    from shared import set_page, inject_theme, load_artifacts, load_training_data, decode_labels

set_page("📈 Analytics • Disease Predictor", "🧬")
inject_theme()

artifacts = load_artifacts()
features = artifacts["features"]
label_enc = artifacts["label_encoder"]
X_train, y_train, X_valid, y_valid, X_all = load_training_data(features)

st.markdown("<h1 class='neon'>📈 Analytics</h1>", unsafe_allow_html=True)

# Overview metrics
try:
    y_all = None
    if y_train is not None and y_valid is not None:
        y_all = pd.concat([y_train, y_valid], ignore_index=True)
    else:
        y_all = y_train if y_train is not None else y_valid
except Exception:
    y_all = y_train if y_train is not None else y_valid

if isinstance(X_all, pd.DataFrame) and y_all is not None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='glass'><h3>📊 Total Records</h3><h2>{len(X_all):,}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='glass'><h3>🧩 Features</h3><h2>{len(features)}</h2></div>", unsafe_allow_html=True)
    with col3:
        unique_diseases = len(y_all.unique()) if y_all is not None else 0
        st.markdown(f"<div class='glass'><h3>🦠 Disease Types</h3><h2>{unique_diseases}</h2></div>", unsafe_allow_html=True)
    with col4:
        avg_symptoms = X_all.mean().mean() if isinstance(X_all, pd.DataFrame) else 0
        st.markdown(f"<div class='glass'><h3>📈 Avg Symptoms</h3><h2>{avg_symptoms:.1f}</h2></div>", unsafe_allow_html=True)

# Interactive filters
st.markdown("<h3 class='section'>Interactive Analysis</h3>", unsafe_allow_html=True)
filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])

with filter_col1:
    chart_type = st.selectbox("Chart Type", ["Bar", "Pie", "Scatter", "Heatmap"])
with filter_col2:
    top_n = st.slider("Top N Items", 5, 50, 20)
with filter_col3:
    show_percentages = st.toggle("Show Percentages", value=True)

# Label distribution (y_all already defined above)

if y_all is not None:
    y_decoded = decode_labels(y_all, label_enc)
    try:
        counts = y_decoded.value_counts().reset_index()
        counts.columns = ['Disease', 'Count']
        
        # Interactive chart based on selection
        if chart_type == "Bar":
            fig = px.bar(counts.head(top_n), x='Disease', y='Count', title=f'Top {top_n} Disease Distribution')
        elif chart_type == "Pie":
            fig = px.pie(counts.head(top_n), values='Count', names='Disease', title=f'Top {top_n} Disease Distribution')
        elif chart_type == "Scatter":
            fig = px.scatter(counts.head(top_n), x='Disease', y='Count', size='Count', title=f'Top {top_n} Disease Distribution')
        else:  # Heatmap
            # Create a simple heatmap representation
            fig = go.Figure(data=go.Heatmap(
                z=[counts.head(top_n)['Count'].values],
                x=counts.head(top_n)['Disease'].values,
                y=['Frequency'],
                colorscale='Viridis'
            ))
            fig.update_layout(title=f'Top {top_n} Disease Distribution (Heatmap)')
        
        if show_percentages and chart_type != "Heatmap":
            total = counts['Count'].sum()
            fig.update_traces(texttemplate='%{y}<br>%{customdata:.1f}%', 
                            customdata=[(count/total)*100 for count in counts.head(top_n)['Count']])
        
        fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width='stretch')
    except Exception as e:
        st.warning(f"Could not render label distribution: {e}")
else:
    st.info("No labels available for distribution plot.")

# Symptom frequency
if isinstance(X_all, pd.DataFrame) and features:
    try:
        usable = [f for f in features if f in X_all.columns]
        freq = (X_all[usable].mean(numeric_only=True).sort_values(ascending=False)).head(top_n)
        freq_df = freq.reset_index(); freq_df.columns = ['Symptom', 'Frequency']
        
        # Interactive symptom chart
        if chart_type == "Bar":
            fig2 = px.bar(freq_df, x='Symptom', y='Frequency', title=f'Top {top_n} Symptoms by Frequency')
        elif chart_type == "Pie":
            fig2 = px.pie(freq_df, values='Frequency', names='Symptom', title=f'Top {top_n} Symptoms by Frequency')
        elif chart_type == "Scatter":
            fig2 = px.scatter(freq_df, x='Symptom', y='Frequency', size='Frequency', title=f'Top {top_n} Symptoms by Frequency')
        else:  # Heatmap
            fig2 = go.Figure(data=go.Heatmap(
                z=[freq_df['Frequency'].values],
                x=freq_df['Symptom'].values,
                y=['Frequency'],
                colorscale='Plasma'
            ))
            fig2.update_layout(title=f'Top {top_n} Symptoms by Frequency (Heatmap)')
        
        if show_percentages and chart_type != "Heatmap":
            total_freq = freq_df['Frequency'].sum()
            fig2.update_traces(texttemplate='%{y}<br>%{customdata:.1f}%', 
                            customdata=[(freq/total_freq)*100 for freq in freq_df['Frequency']])
        
        fig2.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, width='stretch')
    except Exception as e:
        st.warning(f"Could not render symptom frequencies: {e}")
else:
    st.info("Training features unavailable to compute symptom frequencies.")

# Advanced analytics section
st.markdown("<h3 class='section'>Advanced Analytics</h3>", unsafe_allow_html=True)

if isinstance(X_all, pd.DataFrame) and y_all is not None:
    # Correlation analysis
    numeric_cols = X_all.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        st.markdown("#### 🔗 Feature Correlations")
        corr_matrix = X_all[numeric_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        fig_corr.update_layout(
            title="Feature Correlation Matrix",
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_corr, width='stretch')
    
    # Disease vs symptom analysis
    st.markdown("#### 🎯 Disease vs Symptom Analysis")
    disease_symptom_col1, disease_symptom_col2 = st.columns([1, 1])
    
    with disease_symptom_col1:
        selected_disease = st.selectbox("Select Disease", y_decoded.unique() if y_decoded is not None else [])
    with disease_symptom_col2:
        symptom_threshold = st.slider("Symptom Threshold", 0.0, 1.0, 0.3, 0.1)
    
    if selected_disease and isinstance(X_all, pd.DataFrame):
        try:
            disease_mask = y_decoded == selected_disease
            disease_symptoms = X_all[disease_mask].mean()
            top_symptoms = disease_symptoms[disease_symptoms > symptom_threshold].sort_values(ascending=False)
            
            if len(top_symptoms) > 0:
                fig_disease = px.bar(
                    x=top_symptoms.values,
                    y=top_symptoms.index,
                    orientation='h',
                    title=f'Top Symptoms for {selected_disease}',
                    color=top_symptoms.values,
                    color_continuous_scale='Viridis'
                )
                fig_disease.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_disease, width='stretch')
            else:
                st.info(f"No symptoms above threshold {symptom_threshold} for {selected_disease}")
        except Exception as e:
            st.warning(f"Could not analyze disease symptoms: {e}")

# Export functionality
st.markdown("<h3 class='section'>Export Data</h3>", unsafe_allow_html=True)
export_col1, export_col2 = st.columns([1, 1])

with export_col1:
    if st.button("📥 Download Analytics Summary"):
        try:
            summary_data = {
                "total_records": len(X_all) if isinstance(X_all, pd.DataFrame) else 0,
                "total_features": len(features),
                "unique_diseases": len(y_all.unique()) if y_all is not None else 0,
                "avg_symptoms_per_record": float(X_all.mean().mean()) if isinstance(X_all, pd.DataFrame) else 0,
                "most_common_disease": y_decoded.mode().iloc[0] if y_decoded is not None and len(y_decoded) > 0 else "N/A",
                "most_common_symptom": X_all.mean().idxmax() if isinstance(X_all, pd.DataFrame) else "N/A"
            }
            st.download_button(
                label="📊 Download Summary (JSON)",
                data=pd.Series(summary_data).to_json(indent=2),
                file_name="analytics_summary.json",
                mime="application/json"
            )
        except Exception as e:
            st.warning(f"Could not generate summary: {e}")

with export_col2:
    if st.button("📈 Download Charts Data"):
        try:
            if isinstance(X_all, pd.DataFrame) and y_all is not None:
                export_df = X_all.copy()
                export_df['disease'] = y_decoded
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="📋 Download Dataset (CSV)",
                    data=csv_data,
                    file_name="analytics_dataset.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.warning(f"Could not export data: {e}")