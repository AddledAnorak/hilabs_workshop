import os
import glob
import json
import pandas as pd
import plotly.express as px

def generate_report(output_dir="output", report_path="report.md"):
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    if not json_files:
        print("No output JSONs found. Run test.py first.")
        return

    data = []
    for f in json_files:
        with open(f, 'r', encoding='utf-8') as file:
            data.append(json.load(file))

    # Aggregate metrics
    df = pd.json_normalize(data)
    
    # Calculate means for all numeric columns (error rates)
    numeric_cols = df.select_dtypes(include=['number']).columns
    means = df[numeric_cols].mean()

    # Create error heatmap data
    heatmap_data = []
    
    # Entity Type Errors
    entity_cols = [c for c in numeric_cols if c.startswith('entity_type_error_rate.')]
    for c in entity_cols:
        heatmap_data.append({'Category': 'Entity Type', 'Dimension': c.split('.')[-1], 'Error Rate': means[c]})
        
    # Assertion Errors
    assertion_cols = [c for c in numeric_cols if c.startswith('assertion_error_rate.')]
    for c in assertion_cols:
        heatmap_data.append({'Category': 'Assertion', 'Dimension': c.split('.')[-1], 'Error Rate': means[c]})
        
    # Temporality Errors
    temporality_cols = [c for c in numeric_cols if c.startswith('temporality_error_rate.')]
    for c in temporality_cols:
        heatmap_data.append({'Category': 'Temporality', 'Dimension': c.split('.')[-1], 'Error Rate': means[c]})

    # Subject Errors
    subject_cols = [c for c in numeric_cols if c.startswith('subject_error_rate.')]
    for c in subject_cols:
        heatmap_data.append({'Category': 'Subject', 'Dimension': c.split('.')[-1], 'Error Rate': means[c]})

    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Generate Heatmap Plot
    fig = px.density_heatmap(heatmap_df, x="Dimension", y="Category", z="Error Rate", 
                             histfunc="avg", title="Clinical AI Output Error Heatmap",
                             color_continuous_scale="Reds")
    fig.write_image(os.path.join(output_dir, "heatmap.png"))

    # Top weaknesses
    top_weaknesses = heatmap_df.sort_values(by="Error Rate", ascending=False).head(5)

    # Generate Markdown Report
    report_content = f"""# Clinical AI Pipeline Evaluation Report

## Quantitative Evaluation Summary
Analyzed {len(json_files)} patient charts. 

**Overall Pipeline Quality**:
- Average Event Date Accuracy: {means.get('event_date_accuracy', 0):.2%}
- Average Attribute Completeness: {means.get('attribute_completeness', 0):.2%}

## Error Heatmap
![Error Heatmap](output/heatmap.png)

### Error Rates by Dimension

| Category | Dimension | Error Rate |
|----------|-----------|------------|
"""
    for _, row in heatmap_df.sort_values(by=['Category', 'Error Rate'], ascending=[True, False]).iterrows():
        report_content += f"| {row['Category']} | {row['Dimension']} | {row['Error Rate']:.2%} |\n"

    report_content += """
## Top Systemic Weaknesses
"""
    for i, (_, row) in enumerate(top_weaknesses.iterrows(), 1):
        report_content += f"{i}. **{row['Category']} - {row['Dimension']}**: Failed {row['Error Rate']:.2%} of the time.\n"

    report_content += """
## Proposed Guardrails for Improving Reliability
Based on the failure modes observed, the following programmatic guardrails should be implemented in the pipeline:

1. **OCR Artifact Filtering**: Flag or remove entities whose text exactly matches section headers (e.g., "discharge summary", "medication list") or navigational cues ("patient", "encounter_date"). 
2. **Negation Cross-Validation**: If `assertion` == `POSITIVE`, scan the surrounding 5-10 words for negation triggers ("denies", "no", "without", "negative for"). If found, flip assertion to `NEGATIVE` or flag for human review.
3. **Temporal Inconsistency Checks**: If `temporality` == `CURRENT` but the heading contains "History" or text contains "past"/"resolved", flag as temporally ambiguous.
4. **Subject Context Matching**: Ensure entities found under "Family History" sections enforce `subject_error_rate` context bounds (e.g., must be `FAMILY_MEMBER`).
5. **Metadata Completeness Requirements**: Before passing `MEDICINE` entities downstream, validate that required QA attributes (STRENGTH, DOSE, ROUTE) are extracted, and fallback to regex-based parsing if missing.
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"Report generated successfully to {report_path}")

if __name__ == "__main__":
    generate_report()
