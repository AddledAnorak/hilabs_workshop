import streamlit as st
import pandas as pd
import json
import glob
import os
import plotly.express as px

st.set_page_config(page_title="V3 Advanced Clinical AI Evaluator", layout="wide")

st.title("🏥 V3 Evaluation Framework Dashboard")
st.markdown("Advanced analytics powered by **Medical Vocabulary Correlation**, **Cross-File Consensus**, & **Statistical Outliers**.")

@st.cache_data
def load_data():
    # Point to the detailed results for V3 analytics
    json_files = glob.glob(os.path.join("output", "detailed", "*.json"))
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
    st.sidebar.header("🎯 V3 System Overview")
    st.sidebar.metric("Documents Analyzed", len(df))
    st.sidebar.metric("Statistical Anomalies Found", df['file_statistics.is_statistical_anomaly'].sum())
    
    # Calculate totals
    vocab_errs = int(err_df['vocab_correction'].notna().sum()) if 'vocab_correction' in err_df.columns else 0
    coh_warns = df['coherence_statistics.coherence_score'].mean() if 'coherence_statistics.coherence_score' in df.columns else 1.0
    
    st.sidebar.metric("Vocabulary Typing Errors", vocab_errs)
    st.sidebar.metric("Clinical Coherence Score", f"{coh_warns:.1%}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Heatmap & Outliers", "🔍 Entity Drill-Down", "📋 Clinical Context Omissions"])

    with tab1:
        st.subheader("V3 Ensemble Error Rates")
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
            display_cols = [
                'entity', 'type', 'vocab_correction', 'consensus_correction',
                'errors.span_alignment_error', 'errors.clinical_range_error',
                'diagnostic_scores.semantic_confidence', 'diagnostic_scores.grounding', 
                'errors.entity_type_error', 'errors.assertion_error', 'file_name'
            ]
                           
            # Only keep columns that actually exist
            display_cols = [c for c in display_cols if c in view_df.columns]
                           
            st.dataframe(view_df[display_cols], use_container_width=True)
        else:
            st.info("No flagged entities found matching error criteria.")

    with tab3:
        st.subheader("📋 Clinical Context Omissions (Med-Diagnosis Coherence)")
        st.markdown("Entities where a medication was extracted, but its physiological indication (diagnosis) was missed by the pipeline.")
        
        coh_data = []
        for index, row in df.iterrows():
            if 'coherence_statistics.coherence_omission_warnings' in row and isinstance(row['coherence_statistics.coherence_omission_warnings'], list):
                for warn in row['coherence_statistics.coherence_omission_warnings']:
                    coh_data.append({
                        "File": row['file_name'],
                        "Medication Extracted": warn.get('medication'),
                        "Missing Indication (Not Extracted)": ", ".join(warn.get('missing_indication_context', []))
                    })
                    
        if coh_data:
            st.dataframe(pd.DataFrame(coh_data), hide_index=True, use_container_width=True)
        else:
            st.success("No clinical coherence omission warnings found across the batch!")
