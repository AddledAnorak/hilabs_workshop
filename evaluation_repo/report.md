# Clinical AI Pipeline Evaluation Report

## 1. Quantitative Evaluation Summary
This report aggregates the rule-based evaluation results across 30 processed medical charts. The metrics represent average error rates (or inaccuracy rates) bounded between 0.0 and 1.0.

- **Event Date Accuracy (Error Rate)**: 0.1553 (Avg rate of potential date ambiguities)
- **Attribute Completeness**: 0.9267 (Avg rate of entities possessing requested QA metadata)

## 2. Error Heat-Map
| Category | Sub-Category | Error Rate (Avg) | Heat Level |
|----------|-------------|------------------|------------|
| Entity | MEDICINE | 0.0003 | 🟢 Low |
| Entity | PROBLEM | 0.0000 | 🟢 Low |
| Entity | PROCEDURE | 0.0061 | 🟢 Low |
| Entity | TEST | 0.0014 | 🟢 Low |
| Entity | VITAL_NAME | 0.1037 | 🟡 Medium |
| Entity | IMMUNIZATION | 0.0000 | 🟢 Low |
| Entity | MEDICAL_DEVICE | 0.0000 | 🟢 Low |
| Entity | MENTAL_STATUS | 0.0000 | 🟢 Low |
| Entity | SDOH | 0.0000 | 🟢 Low |
| Entity | SOCIAL_HISTORY | 0.0015 | 🟢 Low |
| Assertion | POSITIVE | 0.3200 | 🔴 Severe |
| Assertion | NEGATIVE | 0.1529 | 🟠 High |
| Assertion | UNCERTAIN | 0.8614 | 🔴 Severe |
| Temporality | CURRENT | 0.0000 | 🟢 Low |
| Temporality | CLINICAL_HISTORY | 0.2793 | 🟠 High |
| Temporality | UPCOMING | 0.5686 | 🔴 Severe |
| Temporality | UNCERTAIN | 0.0000 | 🟢 Low |
| Subject | PATIENT | 0.0514 | 🟡 Medium |
| Subject | FAMILY_MEMBER | 0.0174 | 🟢 Low |

## 3. Top Systemic Weaknesses
Based on the error heuristic analysis, the pipeline struggles most with:
1. **Assertion Error: UNCERTAIN** (Error Rate: 0.8614): The system frequently struggles to accurately determine or ground this attribute.
2. **Temporality Error: UPCOMING** (Error Rate: 0.5686): The system frequently struggles to accurately determine or ground this attribute.
3. **Assertion Error: POSITIVE** (Error Rate: 0.3200): The system frequently struggles to accurately determine or ground this attribute.
4. **Temporality Error: CLINICAL_HISTORY** (Error Rate: 0.2793): The system frequently struggles to accurately determine or ground this attribute.
5. **Event Date (Inaccuracy)** (Error Rate: 0.1553): The system frequently struggles to accurately determine or ground this attribute.

## 4. Proposed Guardrails for Improving Reliability
To improve the trustworthiness of the structured extraction pipeline, the following guardrails are recommended:

1. **Entity Grounding Validator**: Implement a lightweight substring matching or embedding similarity check to ensure extracted entities exist in the source transcript. If an entity is hallucinated contextually without text support, flag it for human review.
2. **Negation Confidence Filter**: Use a specialized negation detection model (like Negex) to post-process `POSITIVE` assertions. If Negex predicts negation around a `POSITIVE` entity, downgrade its confidence and require review.
3. **Temporality Heuristics Layer**: Introduce simple keyword-based rules prior to LLM extraction that checks the section header (e.g., 'Past Medical History' vs 'Plan'). If the LLM predicts `CURRENT` under 'Past Medical History', trigger a conflict flag.
4. **Schema Enforcement and Completeness**: Enforce strict schema validation on extracted JSONs. If key clinical entities (e.g., Medication) return without required attributes (dosage, route), the pipeline should loop back to the model with a refinement prompt focusing solely on missing attributes.
5. **Subject Coreference Resolution**: Run a standard Coref model before or alongside the clinical NLP layer to resolve pronouns back to family members, preventing misattribution in heavily descriptive social history narratives.
