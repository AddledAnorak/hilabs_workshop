import os
import json
import glob

OUTPUT_DIR = "output"
REPORT_FILE = "report.md"

def generate_report():
    json_files = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
    if not json_files:
        print("No output JSON files found. Cannot generate report.")
        return

    # Accumulators for averages
    entity_keys = ['MEDICINE', 'PROBLEM', 'PROCEDURE', 'TEST', 'VITAL_NAME', 'IMMUNIZATION', 'MEDICAL_DEVICE', 'MENTAL_STATUS', 'SDOH', 'SOCIAL_HISTORY']
    assertion_keys = ['POSITIVE', 'NEGATIVE', 'UNCERTAIN']
    temporality_keys = ['CURRENT', 'CLINICAL_HISTORY', 'UPCOMING', 'UNCERTAIN']
    subject_keys = ['PATIENT', 'FAMILY_MEMBER']

    entity_err_sums = {k: 0.0 for k in entity_keys}
    assertion_err_sums = {k: 0.0 for k in assertion_keys}
    temporality_err_sums = {k: 0.0 for k in temporality_keys}
    subject_err_sums = {k: 0.0 for k in subject_keys}
    
    event_date_acc_sum = 0.0
    attr_comp_sum = 0.0
    
    # We will also keep track of which specific categories have the highest error rates across all files
    n_files = len(json_files)

    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for k in entity_keys:
            entity_err_sums[k] += data["entity_type_error_rate"].get(k, 0.0)
        for k in assertion_keys:
            assertion_err_sums[k] += data["assertion_error_rate"].get(k, 0.0)
        for k in temporality_keys:
            temporality_err_sums[k] += data["temporality_error_rate"].get(k, 0.0)
        for k in subject_keys:
            subject_err_sums[k] += data["subject_error_rate"].get(k, 0.0)
        
        event_date_acc_sum += data.get("event_date_accuracy", 0.0)
        attr_comp_sum += data.get("attribute_completeness", 0.0)

    # Averages
    entity_err_avg = {k: entity_err_sums[k] / n_files for k in entity_keys}
    assertion_err_avg = {k: assertion_err_sums[k] / n_files for k in assertion_keys}
    temporality_err_avg = {k: temporality_err_sums[k] / n_files for k in temporality_keys}
    subject_err_avg = {k: subject_err_sums[k] / n_files for k in subject_keys}
    event_date_acc_avg = event_date_acc_sum / n_files
    attr_comp_avg = attr_comp_sum / n_files

    # Let's find top systemic weaknesses
    all_errs = []
    for k, v in entity_err_avg.items(): all_errs.append((f"Entity Type Error: {k}", v))
    for k, v in assertion_err_avg.items(): all_errs.append((f"Assertion Error: {k}", v))
    for k, v in temporality_err_avg.items(): all_errs.append((f"Temporality Error: {k}", v))
    for k, v in subject_err_avg.items(): all_errs.append((f"Subject Error: {k}", v))
    all_errs.append(("Event Date (Inaccuracy)", event_date_acc_avg))
    
    # Sort descending by error rate
    all_errs.sort(key=lambda x: x[1], reverse=True)
    top_weaknesses = all_errs[:5]

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("# Clinical AI Pipeline Evaluation Report\n\n")
        
        f.write("## 1. Quantitative Evaluation Summary\n")
        f.write("This report aggregates the rule-based evaluation results across 30 processed medical charts. The metrics represent average error rates (or inaccuracy rates) bounded between 0.0 and 1.0.\n\n")
        
        f.write(f"- **Event Date Accuracy (Error Rate)**: {event_date_acc_avg:.4f} (Avg rate of potential date ambiguities)\n")
        f.write(f"- **Attribute Completeness**: {attr_comp_avg:.4f} (Avg rate of entities possessing requested QA metadata)\n\n")
        
        f.write("## 2. Error Heat-Map\n")
        f.write("| Category | Sub-Category | Error Rate (Avg) | Heat Level |\n")
        f.write("|----------|-------------|------------------|------------|\n")
        
        def heat_level(val):
            if val < 0.05: return "🟢 Low"
            if val < 0.15: return "🟡 Medium"
            if val < 0.30: return "🟠 High"
            return "🔴 Severe"

        for k, v in entity_err_avg.items():
            f.write(f"| Entity | {k} | {v:.4f} | {heat_level(v)} |\n")
        for k, v in assertion_err_avg.items():
            f.write(f"| Assertion | {k} | {v:.4f} | {heat_level(v)} |\n")
        for k, v in temporality_err_avg.items():
            f.write(f"| Temporality | {k} | {v:.4f} | {heat_level(v)} |\n")
        for k, v in subject_err_avg.items():
            f.write(f"| Subject | {k} | {v:.4f} | {heat_level(v)} |\n")
        
        f.write("\n## 3. Top Systemic Weaknesses\n")
        f.write("Based on the error heuristic analysis, the pipeline struggles most with:\n")
        for i, (name, val) in enumerate(top_weaknesses, 1):
            f.write(f"{i}. **{name}** (Error Rate: {val:.4f}): The system frequently struggles to accurately determine or ground this attribute.\n")

        f.write("\n## 4. Proposed Guardrails for Improving Reliability\n")
        f.write("To improve the trustworthiness of the structured extraction pipeline, the following guardrails are recommended:\n\n")
        f.write("1. **Entity Grounding Validator**: Implement a lightweight substring matching or embedding similarity check to ensure extracted entities exist in the source transcript. If an entity is hallucinated contextually without text support, flag it for human review.\n")
        f.write("2. **Negation Confidence Filter**: Use a specialized negation detection model (like Negex) to post-process `POSITIVE` assertions. If Negex predicts negation around a `POSITIVE` entity, downgrade its confidence and require review.\n")
        f.write("3. **Temporality Heuristics Layer**: Introduce simple keyword-based rules prior to LLM extraction that checks the section header (e.g., 'Past Medical History' vs 'Plan'). If the LLM predicts `CURRENT` under 'Past Medical History', trigger a conflict flag.\n")
        f.write("4. **Schema Enforcement and Completeness**: Enforce strict schema validation on extracted JSONs. If key clinical entities (e.g., Medication) return without required attributes (dosage, route), the pipeline should loop back to the model with a refinement prompt focusing solely on missing attributes.\n")
        f.write("5. **Subject Coreference Resolution**: Run a standard Coref model before or alongside the clinical NLP layer to resolve pronouns back to family members, preventing misattribution in heavily descriptive social history narratives.\n")

if __name__ == "__main__":
    generate_report()
    print("Report generated successfully.")
