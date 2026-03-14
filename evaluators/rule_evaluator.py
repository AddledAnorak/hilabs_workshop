import re
import math
from collections import Counter
from evaluators.utils import has_negation_cues, has_historical_cues, has_upcoming_cues, has_family_cues

# More structured heading mappings
HEADING_TYPE_MAP = {
    "medications": ["MEDICINE"],
    "discharge medication": ["MEDICINE"],
    "current meds": ["MEDICINE"],
    "active problems": ["PROBLEM"],
    "principal problem": ["PROBLEM"],
    "social history": ["SOCIAL_HISTORY", "SDOH"],
    "family history": ["PROBLEM", "MEDICINE"],  # Often family diseases
    "vitals": ["VITAL_NAME"],
    "vital signs": ["VITAL_NAME"]
}

def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def find_negation_scope(entity: str, text: str) -> bool:
    """
    NegEx-style: find if entity falls within 5 words of a negation trigger.
    """
    text = text.lower()
    entity = entity.lower()
    triggers = ['no', 'not', 'denies', 'without', 'absent', 'negative for', 'resolved', 'ruled out']
    
    # Locate entity
    try:
        ent_idx = text.index(entity)
    except ValueError:
        return False # If entity isn't even in text exactly, fallback to old logic
        
    words_before = text[:ent_idx].split()[-5:] # up to 5 words before
    scope_text = " ".join(words_before)
    
    return any(t in scope_text for t in triggers)

def evaluate_entity(entity_data: dict) -> dict:
    entity = entity_data.get('entity', '').lower()
    entity_type = entity_data.get('entity_type', '')
    assertion = entity_data.get('assertion', '')
    temporality = entity_data.get('temporality', '')
    subject = entity_data.get('subject', '')
    text = entity_data.get('text', '')
    heading = entity_data.get('heading', '').lower()
    metadata = entity_data.get('metadata_from_qa', {})
    
    errors = {
        'entity_type_error': False,
        'assertion_error': False,
        'temporality_error': False,
        'subject_error': False,
        'event_date_accuracy': 1.0,
        'attribute_completeness': 1.0
    }

    # 1. Entity Type Validation (OCR checks + Heading Context)
    
    # OCR Artifact check (high entropy / gibberish)
    if len(entity) > 10 and entropy(entity) > 4.5: # Highly random string
        errors['entity_type_error'] = True
        
    artifacts = ["mrn", "agmc admit notice", "internally validated risk model",
                 "follow-up appointment", "future appointments", "discharge summary signed", 
                 "encounter_date", "patient", "readmission risk score", "admission information", 
                 "discharge disposition", "medication list", "start taking these medications", "encounter"]
    if any(art in entity for art in artifacts):
        errors['entity_type_error'] = True
    
    if "exacerbation" in entity and entity_type == "MEDICAL_DEVICE":
        errors['entity_type_error'] = True
        
    # Heading Context consistency
    for mapping_key, allowed_types in HEADING_TYPE_MAP.items():
        if mapping_key in heading:
            if entity_type not in allowed_types:
                # If it's physically in the 'medications' list but tagged as VITAL_NAME
                errors['entity_type_error'] = True
            break
        
    # 2. Assertion Validation (NegEx scope)
    # If POSITIVE, but there is a negation trigger right before it
    if assertion == "POSITIVE" and find_negation_scope(entity, text):
        errors['assertion_error'] = True
    # If NEGATIVE, but there are no negation cues anywhere near it
    elif assertion == "NEGATIVE" and not has_negation_cues(text):
        errors['assertion_error'] = True

    # 3. Temporality Validation
    if temporality == "CURRENT" and (has_historical_cues(text) or has_upcoming_cues(text)):
        # Check heading too
        if 'history' in heading:
             errors['temporality_error'] = True
        else:
             errors['temporality_error'] = True
    elif temporality == "UPCOMING" and has_historical_cues(text):
        errors['temporality_error'] = True
    elif temporality == "CLINICAL_HISTORY" and has_upcoming_cues(text):
        errors['temporality_error'] = True

    # 4. Subject Validation
    is_family_context = has_family_cues(text, heading)
    if subject == "PATIENT" and 'family history' in heading:
        errors['subject_error'] = True
    elif subject == "FAMILY_MEMBER" and not is_family_context:
        errors['subject_error'] = True

    # 5. Date Accuracy
    relations = metadata.get('relations', [])
    for rel in relations:
        if 'date' in rel.get('entity_type', ''):
            date_val = rel.get('entity', '')
            if '[' in date_val and ']' in date_val:
                errors['event_date_accuracy'] = 0.0
                
        # 5b. Span Overlap Validation check
        # An entity value's span should reasonably correspond to its extracted text length
        span = rel.get('entity_span', {})
        start, end = span.get('start'), span.get('end')
        ent_text = rel.get('entity', '')
        if start is not None and end is not None and ent_text:
            if abs((end - start) - len(ent_text)) > (len(ent_text) * 0.5):
                # The extracted span is massively longer/shorter than the actual text length
                # E.g. extracted text is "d5w" (3 chars) but span is [40, 60] (20 chars)
                errors['span_alignment_error'] = True
            
    # 6. Attribute Completeness
    if entity_type == "MEDICINE":
        expected_types = {"STRENGTH", "UNIT", "DOSE", "ROUTE", "FREQUENCY", "FORM"}
        found_types = {rel.get("entity_type") for rel in relations}
        if expected_types:
            completeness = len(found_types.intersection(expected_types)) / len(expected_types)
            errors['attribute_completeness'] = completeness
    elif entity_type in ["TEST", "VITAL_NAME"]:
        expected_types = {"TEST_VALUE"} if entity_type == "TEST" else {"VITAL_NAME_VALUE"}
        found_types = {rel.get("entity_type") for rel in relations}
        if expected_types:
            completeness = len(found_types.intersection(expected_types)) / len(expected_types)
            errors['attribute_completeness'] = completeness

    return errors
