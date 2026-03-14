from evaluators.utils import has_negation_cues, has_historical_cues, has_upcoming_cues, has_family_cues

def evaluate_entity(entity_data: dict) -> dict:
    entity = entity_data.get('entity', '').lower()
    entity_type = entity_data.get('entity_type', '')
    assertion = entity_data.get('assertion', '')
    temporality = entity_data.get('temporality', '')
    subject = entity_data.get('subject', '')
    text = entity_data.get('text', '')
    heading = entity_data.get('heading', '')
    metadata = entity_data.get('metadata_from_qa', {})
    
    errors = {
        'entity_type_error': False,
        'assertion_error': False,
        'temporality_error': False,
        'subject_error': False,
        'event_date_accuracy': 1.0,  # 1.0 means no error or N/A, 0.0 means error
        'attribute_completeness': 1.0
    }

    # 1. Entity Type Validation
    # Flag known OCR artifacts or headings
    artifacts = ["mrn", "agmc admit notice", "internally validated risk model",
                 "follow-up appointment", "future appointments", "discharge summary signed", 
                 "encounter_date", "patient", "readmission risk score", "admission information", 
                 "discharge disposition", "medication list", "start taking these medications"]
    if any(art in entity for art in artifacts):
        errors['entity_type_error'] = True
    
    if "exacerbation" in entity and entity_type == "MEDICAL_DEVICE":
        errors['entity_type_error'] = True
        
    # 2. Assertion Validation
    if assertion == "POSITIVE" and has_negation_cues(text):
        errors['assertion_error'] = True
    elif assertion == "NEGATIVE" and not has_negation_cues(text):
        # Could be an error, but let's be conservative
        pass

    # 3. Temporality Validation
    if temporality == "CURRENT" and (has_historical_cues(text) or has_upcoming_cues(text)):
        # If text clearly shows past or future but it's marked current
        errors['temporality_error'] = True
    elif temporality == "UPCOMING" and has_historical_cues(text):
        errors['temporality_error'] = True
    elif temporality == "CLINICAL_HISTORY" and has_upcoming_cues(text):
        errors['temporality_error'] = True

    # 4. Subject Validation
    if subject == "PATIENT" and has_family_cues(text, heading):
        errors['subject_error'] = True
    elif subject == "FAMILY_MEMBER" and not has_family_cues(text, heading):
        errors['subject_error'] = True

    # 5. Date Accuracy
    relations = metadata.get('relations', [])
    for rel in relations:
        if rel.get('entity_type') == 'exact_date' or rel.get('entity_type') == 'derived_date':
            date_val = rel.get('entity', '')
            # If date is placeholder like [encounter_date] instead of real date
            if '[' in date_val and ']' in date_val:
                errors['event_date_accuracy'] = 0.0
            
    # 6. Attribute Completeness
    if entity_type == "MEDICINE":
        # Expect STRENGTH, UNIT, DOSE, ROUTE, FREQUENCY, FORM
        expected_types = {"STRENGTH", "UNIT", "DOSE", "ROUTE", "FREQUENCY", "FORM"}
        found_types = {rel.get("entity_type") for rel in relations}
        if expected_types:
            completeness = len(found_types.intersection(expected_types)) / len(expected_types)
            errors['attribute_completeness'] = completeness
    elif entity_type in ["TEST", "VITAL_NAME"]:
        if entity_type == "TEST":
            expected_types = {"TEST_VALUE"}
        else:
            expected_types = {"VITAL_NAME_VALUE"}
            
        found_types = {rel.get("entity_type") for rel in relations}
        if expected_types:
            completeness = len(found_types.intersection(expected_types)) / len(expected_types)
            errors['attribute_completeness'] = completeness

    return errors
