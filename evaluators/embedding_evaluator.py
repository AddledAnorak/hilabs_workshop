from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from evaluators.vocabulary_evaluator import VOCAB

# Offline fallback for embedding semantic matching
# Uses TF-IDF cosine similarity against the canonical vocabulary.
# This solves the SentenceTransformer SSL download hanging issue.

_vectorizer = None
_reference_vectors = {}

def init_offline_embeddings():
    global _vectorizer, _reference_vectors
    if _vectorizer is not None:
        return
        
    _vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    
    # Build a corpus from all known vocab terms to train the vectorizer space
    all_terms = []
    for cat_list in VOCAB.values():
        all_terms.extend(cat_list)
        
    if not all_terms:
        # Failsafe if vocab wasn't loaded
        return
        
    _vectorizer.fit(all_terms)
    
    # Pre-compute vectors for each category
    for cat_name, terms in VOCAB.items():
        _reference_vectors[cat_name] = _vectorizer.transform(terms)

def evaluate_entity_semantics_offline(entity: str, assigned_type: str) -> dict:
    """
    Evaluates if an extracted entity string semantically matches its assigned type 
    using offline TF-IDF character n-grams trained on the medical vocabulary.
    Returns diagnostic scores and error flags.
    """
    if not entity or not assigned_type:
        return {}
        
    init_offline_embeddings()
    if not _vectorizer or not _reference_vectors:
        return {}

    # Map the pipeline entity types to our vocab categories
    type_map = {
        "MEDICINE": "drug_names",
        "PROBLEM": "conditions",
        "PROCEDURE": "procedures",
        "MEDICAL_DEVICE": "devices",
        "VITAL_NAME": "vitals",
        "TEST": "tests"
    }
    
    expected_cat = type_map.get(assigned_type)
    
    # If it's a type we don't have embeddings for (like SDOH), skip
    if not expected_cat or expected_cat not in _reference_vectors:
        return {}
        
    ent_vec = _vectorizer.transform([entity.lower()])
    
    # Calculate similarity against its assigned category
    expected_sims = cosine_similarity(ent_vec, _reference_vectors[expected_cat])
    max_expected_sim = expected_sims.max() if expected_sims.size > 0 else 0
    
    # Calculate similarity against all categories to find the best match
    best_cat = expected_cat
    best_sim = max_expected_sim
    
    for cat_name, ref_vecs in _reference_vectors.items():
        if cat_name == expected_cat: continue
        sims = cosine_similarity(ent_vec, ref_vecs)
        max_sim = sims.max() if sims.size > 0 else 0
        if max_sim > best_sim:
            best_sim = max_sim
            best_cat = cat_name
            
    # Reverse map back to pipeline types for reporting
    inv_map = {v: k for k, v in type_map.items()}
    best_type_name = inv_map.get(best_cat, best_cat)
            
    results = {
        "diagnostic_scores": {
            "semantic_confidence": float(max_expected_sim)
        }
    }
    
    # If the similarity to its assigned type is extremely low, 
    # AND it matches a different type significantly better (>0.2 margin)
    if max_expected_sim < 0.2 and (best_sim - max_expected_sim) > 0.2:
        results["errors"] = {
            "semantic_mismatch_error": True,
            "best_matching_type": best_type_name
        }
        
    return results
