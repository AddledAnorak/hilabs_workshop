import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_report(output_dir="output", report_path="report.md"):
    # Point to the detailed results so V3 metrics are still captured
    json_files = glob.glob(os.path.join(output_dir, "detailed", "*.json"))
    if not json_files:
        print("No output JSONs found. Run test.py first.")
        return

    data = []
    coherence_warnings_total = 0
    vocab_errors_total = 0
    cross_file_errors_total = 0
    range_errors_total = 0
    span_errors_total = 0
    
    for f in json_files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                j = json.load(file)
                data.append(j)
                
                # Count V3 explicit errors from the entity details log
                for err in j.get('error_details', []):
                    errs = err.get('errors', {})
                    if err.get('vocab_correction'): vocab_errors_total += 1
                    if err.get('consensus_correction'): cross_file_errors_total += 1
                    if errs.get('span_alignment_error'): span_errors_total += 1
                    if errs.get('clinical_range_error'): range_errors_total += 1
                    
                # Coherence
                coh = j.get('coherence_statistics', {})
                coherence_warnings_total += len(coh.get('coherence_omission_warnings', []))
                
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    # Aggregate metrics
    df = pd.json_normalize(data)
    
    # Calculate means
    numeric_cols = [c for c in df.columns if '_error_rate.' in c]
    means = df[numeric_cols].mean()

    # Create error heatmap data
    heatmap_data = []
    
    for c in numeric_cols:
        parts = c.split('_error_rate.')
        if len(parts) == 2:
            cat = parts[0].replace('_', ' ').title()
            dim = parts[1]
            heatmap_data.append({'Category': cat, 'Dimension': dim, 'Error Rate': means[c]})

    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Generate Seaborn Heatmap
    if not heatmap_df.empty:
        pivot_df = heatmap_df.pivot(index="Category", columns="Dimension", values="Error Rate").fillna(0)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_df, annot=True, fmt=".1%", cmap="Reds", cbar_kws={'label': 'Error Rate'})
        plt.title('Clinical AI Pipeline: Error Rates by Dimension', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap.png"), dpi=300)
        plt.close()
        
    # Anomaly Plot
    if 'file_statistics.duplicate_rate' in df.columns and 'file_statistics.is_statistical_anomaly' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, 
            x='file_statistics.duplicate_rate', 
            y='file_statistics.total_entities',
            hue='file_statistics.is_statistical_anomaly',
            palette={True: 'red', False: 'blue'},
            s=100, alpha=0.7
        )
        plt.title('Isolation Forest Anomalies (File-Level)')
        plt.xlabel('Duplication Rate within File')
        plt.ylabel('Total Entities Extracted')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "anomalies.png"), dpi=300)
        plt.close()

    # Top weaknesses
    top_weaknesses = heatmap_df.sort_values(by="Error Rate", ascending=False).head(5)

    # Generate Markdown Report
    report_content = f"""# Advanced Clinical AI Pipeline Report (V2)

## Executive Summary
Evaluation performed using the **V3 Advanced Framework**, featuring:
- Rule-Based heuristics (NegEx boundaries, structural headers)
- **Offline TF-IDF Medical Vocabulary Embeddings**
- **Explicit Medical Dictionary Validation (RxNorm/ICD-10 mappings)**
- **Cross-File Clinical Consensus checking**
- **Medication-Diagnosis Coherence Scoring**
- Statistical Outlier detection

Analyzed {len(json_files)} patient charts.

**Overall Pipeline Quality**:
- Date Extraction Accuracy: {df.get('event_date_accuracy', pd.Series([0])).mean():.2%}
- QA Attribute Completeness: {df.get('attribute_completeness', pd.Series([0])).mean():.2%}
- Files flagged as Statistical Anomalies: {df.get('file_statistics.is_statistical_anomaly', pd.Series([False])).sum()}

**V3 Advanced Diagnostics**:
- 🚨 **Vocabulary Mismatches Found**: {vocab_errors_total} *(e.g. recognized drug typed as MENTAL_STATUS)*
- 🚨 **Clinical Coherence Omission Flags**: {coherence_warnings_total} *(e.g. Insulin extracted, but no Diabetes diagnosis extracted)*
- 🚨 **Vital/Lab Range Violations**: {range_errors_total} *(e.g. Heart Rate = 3200)*
- 🚨 **Cross-File Consensus Outliers**: {cross_file_errors_total}
- 🚨 **Span Alignment Errors**: {span_errors_total}

---

## 2D Error Matrix
![Error Matrix](output/heatmap.png)

## Isolation Forest Output (Chart-level Outliers)
![Anomalies](output/anomalies.png)

---

## Top Systemic Weaknesses
"""
    for i, (_, row) in enumerate(top_weaknesses.iterrows(), 1):
        report_content += f"{i}. **{row['Category']} - {row['Dimension']}**: Failed {row['Error Rate']:.2%} of the time.\n"

    report_content += """
---
## Proposed Security & Reliability Guardrails
1. **Medical Vocabulary Anchor**: The pipeline fundamentally failed by typing `morphine sulfate` as MENTAL_STATUS and `apixaban` as SDOH. A hard-coded reference check against RxNorm eliminates this entire class of hallucination.
2. **Medication-Diagnosis Coherence Check**: The V3 pipeline identified charts where active diabetic or anticoagulant medications were extracted, but the corresponding diagnosis was completely missed. These clinical omissions are dangerous for downstream analytics and must trigger a secondary extraction pass.
3. **Clinical Range Validation**: Vitals numeric extraction must be hard-capped. A heart rate of 3200 is physiologically impossible and indicates a bounding box or OCR fault.
4. **Duplicate Extractor Limiter**: Abort pipeline if identical contextual triplets occur more than 5 times in a single chart.

*Generated by V3 Evaluation Ensemble Engine.*
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"V2 Report generated successfully to {report_path}")

if __name__ == "__main__":
    generate_report()
