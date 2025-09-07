import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
try:
    from ..shared import set_page, inject_theme, load_artifacts, load_training_data, ensure_arrow_compatibility, safe_dataframe_display
except Exception:
    from shared import set_page, inject_theme, load_artifacts, load_training_data, ensure_arrow_compatibility, safe_dataframe_display

set_page("📊 Data Explorer • Disease Predictor", "🧬")
inject_theme()

artifacts = load_artifacts()
features = artifacts["features"]
X_train, y_train, X_valid, y_valid, X_all = load_training_data(features)

st.markdown("<h1 class='neon'>📊 Data Explorer</h1>", unsafe_allow_html=True)

# Overview metrics
if isinstance(X_all, pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='glass'><h3>📊 Total Records</h3><h2>{len(X_all):,}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='glass'><h3>🧩 Features</h3><h2>{len(features)}</h2></div>", unsafe_allow_html=True)
    with col3:
        missing_pct = (X_all.isnull().sum().sum() / (X_all.shape[0] * X_all.shape[1])) * 100
        st.markdown(f"<div class='glass'><h3>❌ Missing Data</h3><h2>{missing_pct:.1f}%</h2></div>", unsafe_allow_html=True)
    with col4:
        numeric_cols = len(X_all.select_dtypes(include=[np.number]).columns)
        st.markdown(f"<div class='glass'><h3>🔢 Numeric Features</h3><h2>{numeric_cols}</h2></div>", unsafe_allow_html=True)

# Interactive filters
st.markdown("<h3 class='section'>Interactive Data Exploration</h3>", unsafe_allow_html=True)
filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])

with filter_col1:
    view_type = st.selectbox("View Type", ["Overview", "Detailed", "Custom Filter"])
with filter_col2:
    max_rows = st.slider("Max Rows to Display", 5, 100, 20)
with filter_col3:
    show_stats = st.toggle("Show Statistics", value=True)

if isinstance(X_all, pd.DataFrame):
    # Data sample based on view type
    if view_type == "Overview":
        st.markdown("<h3 class='section'>Dataset Overview</h3>", unsafe_allow_html=True)
        safe_dataframe_display(X_all.head(max_rows), "Dataset Sample")
    elif view_type == "Detailed":
        st.markdown("<h3 class='section'>Detailed View</h3>", unsafe_allow_html=True)
        safe_dataframe_display(X_all.head(max_rows), "Dataset Sample")
        
        # Show data quality metrics
        st.markdown("#### 📊 Data Quality Metrics")
        quality_col1, quality_col2 = st.columns([1, 1])
        
        with quality_col1:
            # Missing values heatmap
            missing_data = X_all.isnull().sum()
            if missing_data.sum() > 0:
                fig_missing = px.bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h',
                    title="Missing Values by Feature",
                    color=missing_data.values,
                    color_continuous_scale="Viridis"
                )
                fig_missing.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_missing, width='stretch')
            else:
                st.success("✅ No missing values detected")
        
        with quality_col2:
            # Data types distribution
            dtype_counts = X_all.dtypes.value_counts()
            fig_dtypes = px.pie(
                values=dtype_counts.values,
                names=dtype_counts.index,
                title="Feature Data Types"
            )
            fig_dtypes.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_dtypes, width='stretch')
    
    else:  # Custom Filter
        st.markdown("<h3 class='section'>Custom Data Filter</h3>", unsafe_allow_html=True)
        
        # Column selection
        selected_cols = st.multiselect(
            "Select columns to display:",
            options=X_all.columns.tolist(),
            default=X_all.columns.tolist()[:min(5, len(X_all.columns))]
        )
        
        if selected_cols:
            filtered_data = X_all[selected_cols]
            
            # Row filtering
            filter_col1, filter_col2 = st.columns([1, 1])
            
            with filter_col1:
                numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    selected_numeric = st.selectbox("Filter by numeric column:", [""] + numeric_cols.tolist())
                    if selected_numeric:
                        min_val = float(filtered_data[selected_numeric].min())
                        max_val = float(filtered_data[selected_numeric].max())
                        range_val = st.slider(
                            f"{selected_numeric} Range",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val)
                        )
                        filtered_data = filtered_data[
                            (filtered_data[selected_numeric] >= range_val[0]) & 
                            (filtered_data[selected_numeric] <= range_val[1])
                        ]
            
            with filter_col2:
                categorical_cols = filtered_data.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    selected_cat = st.selectbox("Filter by categorical column:", [""] + categorical_cols.tolist())
                    if selected_cat:
                        unique_values = filtered_data[selected_cat].unique()
                        selected_values = st.multiselect(
                            f"Select {selected_cat} values:",
                            options=unique_values,
                            default=unique_values
                        )
                        if selected_values:
                            filtered_data = filtered_data[filtered_data[selected_cat].isin(selected_values)]
            
            st.markdown(f"**Filtered Results:** {len(filtered_data)} rows")
            safe_dataframe_display(filtered_data.head(max_rows), "Filtered Data")
    
    # Statistics section
    if show_stats:
        st.markdown("<h3 class='section'>Statistical Summary</h3>", unsafe_allow_html=True)
        try:
            stats_col1, stats_col2 = st.columns([1, 1])
            
            with stats_col1:
                safe_dataframe_display(X_all.describe(), "Numeric Statistics")
            
            with stats_col2:
                # Custom statistics
                custom_stats = {
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        X_all.mean().mean(),
                        X_all.median().median(),
                        X_all.std().mean(),
                        X_all.min().min(),
                        X_all.max().max()
                    ]
                }
                custom_df = pd.DataFrame(custom_stats)
                safe_dataframe_display(custom_df, "Custom Statistics")
        except Exception:
            pass

    # Data visualization section
    st.markdown("<h3 class='section'>Data Visualization</h3>", unsafe_allow_html=True)
    
    viz_col1, viz_col2 = st.columns([1, 1])
    
    with viz_col1:
        # Feature distribution
        numeric_features = X_all.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 0:
            selected_feature = st.selectbox("Select feature for distribution:", numeric_features)
            if selected_feature:
                fig_dist = px.histogram(
                    X_all,
                    x=selected_feature,
                    nbins=30,
                    title=f"Distribution of {selected_feature}"
                )
                fig_dist.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_dist, width='stretch')
    
    with viz_col2:
        # Correlation heatmap
        if len(numeric_features) > 1:
            corr_matrix = X_all[numeric_features].corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 8}
            ))
            fig_corr.update_layout(
                title="Feature Correlation Matrix",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_corr, width='stretch')
    
    # Export functionality
    st.markdown("<h3 class='section'>Export Data</h3>", unsafe_allow_html=True)
    export_col1, export_col2, export_col3 = st.columns([1, 1, 1])
    
    with export_col1:
        if st.button("📥 Download Sample Data"):
            csv_data = X_all.head(max_rows).to_csv(index=False)
            st.download_button(
                label="📋 Download CSV",
                data=csv_data,
                file_name="data_sample.csv",
                mime="text/csv"
            )
    
    with export_col2:
        if st.button("📊 Download Statistics"):
            try:
                stats_data = X_all.describe().to_csv()
                st.download_button(
                    label="📈 Download Stats",
                    data=stats_data,
                    file_name="data_statistics.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.warning(f"Could not generate statistics: {e}")
    
    with export_col3:
        if st.button("🔍 Download Data Info"):
            try:
                info_data = {
                    "shape": X_all.shape,
                    "columns": X_all.columns.tolist(),
                    "dtypes": X_all.dtypes.to_dict(),
                    "missing_values": X_all.isnull().sum().to_dict(),
                    "memory_usage": X_all.memory_usage(deep=True).sum()
                }
                st.download_button(
                    label="📋 Download Info",
                    data=pd.Series(info_data).to_json(indent=2),
                    file_name="data_info.json",
                    mime="application/json"
                )
            except Exception as e:
                st.warning(f"Could not generate info: {e}")
else:
    st.info("No dataset available to display.")