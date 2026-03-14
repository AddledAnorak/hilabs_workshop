import streamlit as st
import pandas as pd
import json
import glob
import os
import plotly.express as px

st.set_page_config(page_title="Clinical AI Evaluator", layout="wide")

st.title("🏥 Clinical AI Pipeline Evaluation Dashboard")
st.markdown("Monitor and analyze output reliability of extracted medical records.")

# Load Data
@st.cache_data
def load_data():
    json_files = glob.glob(os.path.join("output", "*.json"))
    if not json_files:
        return pd.DataFrame()
    data = []
    for f in json_files:
        with open(f, 'r') as file:
            data.append(json.load(file))
    return pd.json_normalize(data)

df = load_data()

if df.empty:
    st.warning("No output data found. Please run the evaluation batch script first.")
    st.code("python test.py --batch")
else:
    # Sidebar metrics
    st.sidebar.header("🎯 System Overview")
    st.sidebar.metric("Documents Analyzed", len(df))
    st.sidebar.metric("Avg Date Accuracy", f"{df['event_date_accuracy'].mean():.1%}")
    st.sidebar.metric("Avg Attribute Completeness", f"{df['attribute_completeness'].mean():.1%}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Aggregate Analytics", "🔍 File-level Inspection", "📋 Guardrails"])

    with tab1:
        st.subheader("Error Heatmap Overview")
        numeric_cols = df.select_dtypes(include=['number']).columns
        means = df[numeric_cols].mean()

        heatmap_data = []
        for c in numeric_cols:
            if '_error_rate.' in c:
                parts = c.split('_error_rate.')
                cat, dim = parts[0], parts[1]
                heatmap_data.append({'Category': cat.replace('_', ' ').title(), 'Dimension': dim, 'Error Rate': means[c]})
        
        heatmap_df = pd.DataFrame(heatmap_data)
        
        if not heatmap_df.empty:
            fig = px.density_heatmap(heatmap_df, x="Dimension", y="Category", z="Error Rate", 
                                    histfunc="avg", text_auto=".1%", 
                                    color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Systemic Weaknesses")
                top_errs = heatmap_df.sort_values("Error Rate", ascending=False).head(5)
                st.dataframe(top_errs.style.format({"Error Rate": "{:.2%}"}), hide_index=True)

    with tab2:
        st.subheader("File-level Inspection")
        selected_file = st.selectbox("Select File", df["file_name"])
        
        file_data = df[df["file_name"] == selected_file].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Event Date Accuracy", f"{file_data['event_date_accuracy']:.1%}")
        with col2:
            st.metric("Attribute Completeness", f"{file_data['attribute_completeness']:.1%}")
            
        st.json(file_data.to_dict())

    with tab3:
        st.subheader("🛡️ Recommended Guardrails")
        st.markdown("""
        **1. OCR Artifact Filtering**: Flag or remove entities whose text exactly matches section headers.
        **2. Negation Cross-Validation**: If `assertion` == `POSITIVE`, scan surrounding context for negation triggers.
        **3. Temporal Inconsistency Checks**: Validated explicitly stated temporal shifts (e.g. "past") against flags.
        **4. Structural Context Match**: Map entity sections (Family History) directly to subject resolution bounds.
        """)
