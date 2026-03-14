import streamlit as st
import pandas as pd
import json
import glob
import os
import plotly.express as px

st.set_page_config(page_title="V2 Advanced Clinical AI Evaluator", layout="wide")

st.title("🏥 V2 Evaluation Framework Dashboard")
st.markdown("Advanced analytics powered by **Sentence-Transformers** & **Isolation Forests**.")

@st.cache_data
def load_data():
    json_files = glob.glob(os.path.join("output", "*.json"))
    if not json_files:
        return pd.DataFrame(), pd.DataFrame()
        
    data = []
    error_logs = []
    
    for f in json_files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                j = json.load(file)
                data.append(j)
                
                # Unroll error details
                for err in j.get('error_details', []):
                    err['file_name'] = j['file_name']
                    error_logs.append(err)
        except:
            pass
            
    return pd.json_normalize(data), pd.json_normalize(error_logs)

df, err_df = load_data()

if df.empty:
    st.warning("No V2 output data found. Please run the evaluation batch script first.")
    st.code("python test.py --batch")
else:
    # Sidebar metrics
    st.sidebar.header("🎯 V2 System Overview")
    st.sidebar.metric("Documents Analyzed", len(df))
    st.sidebar.metric("Statistical Anomalies Found", df['file_statistics.is_statistical_anomaly'].sum())
    st.sidebar.metric("Avg Extraction Redundancy", f"{df['file_statistics.duplicate_rate'].mean():.1%}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Heatmap & Outliers", "🔍 Entity Drill-Down", "📋 Recommended Guardrails"])

    with tab1:
        st.subheader("V2 Ensemble Error Rates")
        numeric_cols = [c for c in df.columns if '_error_rate.' in c]
        means = df[numeric_cols].mean()

        heatmap_data = []
        for c in numeric_cols:
            parts = c.split('_error_rate.')
            if len(parts) == 2:
                cat, dim = parts[0].replace('_', ' ').title(), parts[1]
                heatmap_data.append({'Category': cat, 'Dimension': dim, 'Error Rate': means[c]})
        
        heatmap_df = pd.DataFrame(heatmap_data)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if not heatmap_df.empty:
                fig = px.density_heatmap(heatmap_df, x="Dimension", y="Category", z="Error Rate", 
                                        histfunc="avg", text_auto=".1%", 
                                        color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)
                
        with col2:
            st.subheader("Top Weaknesses")
            if not heatmap_df.empty:
                top_errs = heatmap_df.sort_values("Error Rate", ascending=False).head(5)
                st.dataframe(top_errs.style.format({"Error Rate": "{:.2%}"}), hide_index=True)
                
        st.subheader("Statistical Outlier Detection (Isolation Forest)")
        if 'file_statistics.duplicate_rate' in df.columns:
            fig2 = px.scatter(df, x='file_statistics.duplicate_rate', y='file_statistics.total_entities',
                             color='file_statistics.is_statistical_anomaly',
                             hover_data=['file_name', 'file_statistics.pct_medicine'],
                             title="Document-Level Anomaly Clustering",
                             labels={"file_statistics.duplicate_rate": "Duplication Rate",
                                     "file_statistics.total_entities": "Total Entities",
                                     "file_statistics.is_statistical_anomaly": "Is Anomaly"})
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Entity-Level Log Explorer")
        st.markdown("Dive into entities flagged by the ensemble (Rules + LLM + Embeddings).")
        if not err_df.empty:
            # Filters
            f_col1, f_col2 = st.columns(2)
            sel_type = f_col1.multiselect("Filter by Entity Type", err_df['type'].unique())
            sel_file = f_col2.multiselect("Filter by File", err_df['file_name'].unique())
            
            view_df = err_df.copy()
            if sel_type: view_df = view_df[view_df['type'].isin(sel_type)]
            if sel_file: view_df = view_df[view_df['file_name'].isin(sel_file)]
            
            # Format columns nicely
            display_cols = ['entity', 'type', 'assertion', 'diagnostic_scores.semantic_confidence', 
                           'diagnostic_scores.grounding', 'errors.entity_type_error', 
                           'errors.assertion_error', 'file_name']
                           
            # Only keep columns that actually exist
            display_cols = [c for c in display_cols if c in view_df.columns]
                           
            st.dataframe(view_df[display_cols], use_container_width=True)
        else:
            st.info("No flagged entities found matching error criteria.")

    with tab3:
        st.subheader("🛡️ V2 Recommended Guardrails")
        st.markdown("""
        **1. Vector Embedding Fence**:
        Deploy a Sentence-Transformer filter downstream. Entities dropping below 0.15 cosine similarity against their canonical `entity_type` references should trigger human review.
        
        **2. Strict Redundancy Caps**:
        The Isolation Forest identified documents with duplicate extraction rates above 20%. Any identical contextual triplet mapping should abort processing to prevent downstream analytics poisoning.
        
        **3. Contextual Bounds Framing (NegEx)**:
        Relying on regex to find 'denies' anywhere in the raw text caused massive false Positives. Guardrail deployed: Negation cross-validity is now strictly bound to a trailing 5-token `scope window` from the matched entity index. 
        """)
