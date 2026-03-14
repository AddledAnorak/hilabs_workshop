import numpy as np
import warnings
from sentence_transformers import SentenceTransformer

# Suppress HuggingFace warnings
warnings.filterwarnings("ignore")

# Load a small, fast sentence-transformer model
model = None

# Reference canonical sets for different entity types
# This helps us identify if an entity is drastically misclassified
REFERENCE_SETS = {
    "MEDICINE": ["aspirin", "metoprolol", "lisinopril", "ibuprofen", "acetaminophen", "atorvastatin", "insulin"],
    "PROBLEM": ["hypertension", "diabetes", "copd", "heart failure", "asthma", "pneumonia", "anxiety", "pain"],
    "PROCEDURE": ["blood transfusion", "mri scan", "surgery", "biopsy", "x-ray", "colonoscopy", "intubation"],
    "TEST": ["glucose", "hemoglobin", "creatinine", "cholesterol", "white blood count", "potassium"],
    "VITAL_NAME": ["blood pressure", "heart rate", "temperature", "respiratory rate", "oxygen saturation", "weight"],
    "MEDICAL_DEVICE": ["pacemaker", "cpap machine", "insulin pump", "hearing aid", "wheelchair", "catheter", "stent"],
    "IMMUNIZATION": ["flu vaccine", "covid-19 vaccine", "tetanus shot", "pneumococcal vaccine", "shingles vaccine"]
}

# Cache for precomputed embeddings
_reference_embeddings = {}

def get_model():
    global model
    if model is None:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Warning: Failed to load sentence-transformer model: {e}")
            return None
    return model

def precompute_reference_embeddings():
    m = get_model()
    if not m or _reference_embeddings:
        return
        
    for ent_type, examples in REFERENCE_SETS.items():
        _reference_embeddings[ent_type] = m.encode(examples)

def cosine_similarity(a, b):
    # Normalized dot product
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=len(b.shape)-1) + 1e-9)

def evaluate_entity_semantics(entity: str, assigned_type: str) -> dict:
    """
    Checks if the entity text is semantically bizarre for its assigned type.
    Returns dict with error flag and confidence.
    """
    if not entity or assigned_type not in REFERENCE_SETS:
        return {}
        
    m = get_model()
    if not m:
        return {}
        
    precompute_reference_embeddings()
    
    # Encode the target entity
    try:
        ent_emb = m.encode([entity])[0]
    except:
        return {}
        
    scores = {}
    
    # Calculate similarity to all known reference sets
    for ref_type, ref_embs in _reference_embeddings.items():
        # Get max similarity to any example in the reference set
        sims = cosine_similarity(ent_emb, ref_embs)
        scores[ref_type] = float(np.max(sims))
        
    # If the assigned type is significantly lower than the highest matching type, 
    # flag it as a semantic mismatch
    
    actual_score = scores.get(assigned_type, 0.0)
    best_matching_type = max(scores.items(), key=lambda x: x[1])
    best_type, best_score = best_matching_type
    
    # Threshold for misclassification:
    # 1. Matches another category much better (> 0.2 difference)
    # 2. Or is completely orthogonal to its assigned category (< 0.1)
    is_mismatch = False
    
    if best_type != assigned_type and (best_score - actual_score > 0.25):
        if best_score > 0.4: # Only if it strongly matches something else
            is_mismatch = True
            
    if actual_score < 0.15:
        # High likelihood it's an OCR artifact or highly unrelated string
        is_mismatch = True
        
    # Special carve-outs for known edge cases
    if assigned_type == "PROCEDURE" and "appointment" in entity.lower():
        is_mismatch = False # Not clinically a procedure, but often tagged as one
        
    return {
        'semantic_type_error': is_mismatch,
        'semantic_confidence': actual_score,
        'best_matching_type': best_type if is_mismatch else assigned_type
    }
