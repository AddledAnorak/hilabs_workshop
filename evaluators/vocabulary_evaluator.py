import os
from evaluators.utils import load_json

def get_medical_vocabulary():
    """Loads the static medical vocabulary for deterministic type checking."""
    vocab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'medical_vocabulary.json')
    if os.path.exists(vocab_path):
        return load_json(vocab_path)
    return {}

VOCAB = get_medical_vocabulary()

# Map JSON list names to the pipeline's expected entity types
TYPE_MAPPING = {
    "drug_names": "MEDICINE",
    "conditions": "PROBLEM",
    "procedures": "PROCEDURE",
    "devices": "MEDICAL_DEVICE",
    "vitals": "VITAL_NAME",
    "tests": "TEST"
}

def evaluate_entity_vocabulary(entity_data: dict) -> dict:
    """
    Checks if an entity text exactly matches a known medical vocabulary term.
    If it does, but the assigned entity_type contradicts the known type, flag it.
    """
    entity = entity_data.get('entity', '').lower().strip()
    assigned_type = entity_data.get('entity_type', '')
    
    if not entity or not VOCAB:
        return {}
        
    for vocab_list, true_type in TYPE_MAPPING.items():
        if entity in VOCAB.get(vocab_list, []):
            if assigned_type != true_type:
                return {
                    'vocabulary_mismatch_error': True,
                    'vocabulary_correction': true_type
                }
            else:
                return {
                    'vocabulary_match_success': True # Strong signal it's correct
                }
                
    # If we didn't find it, we can't definitively call it an error here
    return {}
