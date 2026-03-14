import json
import sys
import re
import os

def check_keyword_match(text, keywords):
    text_lower = text.lower()
    return any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower) for kw in keywords)

def evaluate_chart(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    entity_counts = {"MEDICINE": 0, "PROBLEM": 0, "PROCEDURE": 0, "TEST": 0, "VITAL_NAME": 0, "IMMUNIZATION": 0, "MEDICAL_DEVICE": 0, "MENTAL_STATUS": 0, "SDOH": 0, "SOCIAL_HISTORY": 0}
    entity_errors = {"MEDICINE": 0, "PROBLEM": 0, "PROCEDURE": 0, "TEST": 0, "VITAL_NAME": 0, "IMMUNIZATION": 0, "MEDICAL_DEVICE": 0, "MENTAL_STATUS": 0, "SDOH": 0, "SOCIAL_HISTORY": 0}

    assertion_counts = {"POSITIVE": 0, "NEGATIVE": 0, "UNCERTAIN": 0}
    assertion_errors = {"POSITIVE": 0, "NEGATIVE": 0, "UNCERTAIN": 0}

    temporality_counts = {"CURRENT": 0, "CLINICAL_HISTORY": 0, "UPCOMING": 0, "UNCERTAIN": 0}
    temporality_errors = {"CURRENT": 0, "CLINICAL_HISTORY": 0, "UPCOMING": 0, "UNCERTAIN": 0}

    subject_counts = {"PATIENT": 0, "FAMILY_MEMBER": 0}
    subject_errors = {"PATIENT": 0, "FAMILY_MEMBER": 0}

    total_entities = len(data)
    missing_attributes = 0
    date_accuracy_issues = 0

    negation_keywords = ['no', 'not', 'denies', 'without', 'negative', 'ruled out', 'rules out']
    uncertain_keywords = ['possible', 'probable', 'suspected', 'suspect', 'unclear', 'maybe', 'rule out', '?']
    history_keywords = ['history', 'hx', 'past', 'ago', 'previous', 'resolved']
    upcoming_keywords = ['plan', 'scheduled', 'return', 'follow up', 'will', 'future', 'tomorrow']
    family_keywords = ['mother', 'father', 'sister', 'brother', 'family', 'fhx', 'aunt', 'uncle', 'grandmother', 'grandfather']

    for item in data:
        entity = item.get('entity', '').lower()
        entity_type = item.get('entity_type', '')
        assertion = item.get('assertion', '')
        temporality = item.get('temporality', '')
        subject = item.get('subject', '')
        text = item.get('text', '')

        # Fallback tracking logic in case types aren't in our predefined schema initialization
        if entity_type not in entity_counts:
            entity_counts[entity_type] = 0
            entity_errors[entity_type] = 0
            
        entity_counts[entity_type] += 1
        
        if assertion in assertion_counts:
            assertion_counts[assertion] += 1
            
        if temporality in temporality_counts:
            temporality_counts[temporality] += 1
            
        if subject in subject_counts:
            subject_counts[subject] += 1

        # 1. Entity type error: Does the entity text actually exist in the OCR text or has it been hallucinated/garbled?
        if entity not in text.lower():
            # Weak fallback: check if most words from entity exist in text
            entity_words = set(entity.split())
            text_words = set(text.lower().split())
            if not entity_words.issubset(text_words):
                entity_errors[entity_type] += 1

        # 2. Assertion error heuristics
        if assertion == "NEGATIVE" and not check_keyword_match(text, negation_keywords):
            assertion_errors["NEGATIVE"] += 1
            
        if assertion == "UNCERTAIN" and not check_keyword_match(text, uncertain_keywords):
            # Sometimes uncertainty is captured differently, but if no keywords, flag it as possible error
            assertion_errors["UNCERTAIN"] += 1

        if assertion == "POSITIVE" and check_keyword_match(text, negation_keywords):
            # Positive entity but negation words in proximity/text snippet -> likely error
            # Simple heuristic - not perfect, but identifies systemic weaknesses
            assertion_errors["POSITIVE"] += 1

        # 3. Temporality error heuristics
        if temporality == "CLINICAL_HISTORY" and not check_keyword_match(text, history_keywords):
            # Temporality heuristic misses could mean context is implied, but flags a warning
            temporality_errors["CLINICAL_HISTORY"] += 1
            
        if temporality == "UPCOMING" and not check_keyword_match(text, upcoming_keywords):
            temporality_errors["UPCOMING"] += 1

        # 4. Subject error heuristics
        if subject == "FAMILY_MEMBER" and not check_keyword_match(text, family_keywords):
            subject_errors["FAMILY_MEMBER"] += 1
            
        if subject == "PATIENT" and check_keyword_match(text, family_keywords):
            # Text contains family words but attributed to patient. Could be an error (or just mentioning family)
            subject_errors["PATIENT"] += 1

        # 5. Attribute completeness
        # We expect test, vitals, procedures to often have relations (metadata) extracted
        metadata = item.get('metadata_from_qa', {})
        if not metadata and entity_type in ['MEDICINE', 'TEST', 'VITAL_NAME']:
            missing_attributes += 1

        # 6. Event date accuracy
        # Simple heuristic: if entity is in a document snippet with dates, did the system misinterpret it?
        # For simplicity, we flag as error if text has 2+ dates and no clear extraction handles it
        dates_in_text = len(re.findall(r'\[date\]|\[encounter_date\]', text.lower()))
        if dates_in_text > 1 and temporality == "CURRENT":
            # Just a placeholder heuristic for date ambiguity
            date_accuracy_issues += 1

    def calc_rate(err, total):
        return round(err / total, 4) if total > 0 else 0.0

    output_data = {
        "file_name": os.path.basename(input_file),
        "entity_type_error_rate": {
            k: calc_rate(entity_errors.get(k, 0), entity_counts.get(k, 0)) for k in ['MEDICINE', 'PROBLEM', 'PROCEDURE', 'TEST', 'VITAL_NAME', 'IMMUNIZATION', 'MEDICAL_DEVICE', 'MENTAL_STATUS', 'SDOH', 'SOCIAL_HISTORY']
        },
        "assertion_error_rate": {
            k: calc_rate(assertion_errors.get(k, 0), assertion_counts.get(k, 0)) for k in assertion_counts
        },
        "temporality_error_rate": {
            k: calc_rate(temporality_errors.get(k, 0), temporality_counts.get(k, 0)) for k in temporality_counts
        },
        "subject_error_rate": {
            k: calc_rate(subject_errors.get(k, 0), subject_counts.get(k, 0)) for k in subject_counts
        },
        "event_date_accuracy": calc_rate(date_accuracy_issues, total_entities),
        "attribute_completeness": 1.0 - calc_rate(missing_attributes, total_entities) # Higher is better, so 1 - error rate (schema asks for completeness)
    }

    # If any other keys appear in the input data, retain them dynamically in the output as well
    for k, v in entity_counts.items():
        if k not in output_data["entity_type_error_rate"] and k != '':
            output_data["entity_type_error_rate"][k] = calc_rate(entity_errors.get(k, 0), v)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test.py <input.json> <output.json>")
        sys.exit(1)
        
    evaluate_chart(sys.argv[1], sys.argv[2])
