# Simple heuristic map indicating which drugs treat which diagnosis categories 
# (massively simplified for evaluation proof-of-concept)

MED_TO_CONDITION_MAP = {
    # Cardiology / BP
    "metoprolol": ["hypertension", "heart failure", "atrial fibrillation"],
    "lisinopril": ["hypertension", "heart failure"],
    "losartan": ["hypertension", "heart failure"],
    "amlodipine": ["hypertension"],
    "hydrochlorothiazide": ["hypertension"],
    "furosemide": ["heart failure", "hypertension", "edema"],
    "spironolactone": ["heart failure", "hypertension"],
    "sacubitril/valsartan": ["heart failure"],
    
    # Blood thinners
    "apixaban": ["atrial fibrillation", "deep vein thrombosis", "pulmonary embolism"],
    "rivaroxaban": ["atrial fibrillation", "deep vein thrombosis", "pulmonary embolism"],
    "warfarin": ["atrial fibrillation", "deep vein thrombosis", "pulmonary embolism"],
    "clopidogrel": ["coronary artery disease", "myocardial infarction", "stroke"],
    
    # Diabetes
    "metformin": ["diabetes"],
    "insulin": ["diabetes"],
    "empagliflozin": ["diabetes", "heart failure"],
    "glipizide": ["diabetes"],
    
    # Pulmonary
    "albuterol": ["asthma", "copd"],
    "fluticasone": ["asthma", "copd"],
    "montelukast": ["asthma"],
    
    # Lipids
    "atorvastatin": ["hyperlipidemia", "coronary artery disease", "myocardial infarction"],
    "simvastatin": ["hyperlipidemia", "coronary artery disease"],
    "rosuvastatin": ["hyperlipidemia", "coronary artery disease"],
    
    # Pain / Opioids
    "morphine sulfate": ["pain", "chronic pain"],
    "oxycodone": ["pain", "chronic pain"],
    "hydrocodone": ["pain", "chronic pain"],
    "acetaminophen": ["pain", "osteoarthritis"],
    "ibuprofen": ["pain", "osteoarthritis", "rheumatoid arthritis"]
}

def evaluate_chart_coherence(all_entities: list) -> dict:
    """
    Checks if extracted medications make sense given the extracted problems.
    Returns file-level coherence scores and flags missing contexts.
    """
    if not all_entities:
        return {}
        
    meds_found = set()
    problems_found = set()
    
    for ent in all_entities:
        text = ent.get('entity', '').lower()
        ent_type = ent.get('entity_type', '')
        # Only consider active current problems/meds
        if ent.get('assertion') == 'POSITIVE':
            if ent_type == 'MEDICINE':
                meds_found.add(text)
            elif ent_type == 'PROBLEM':
                problems_found.add(text)
                
    coherence_warnings = []
    
    # For every med we found that has known indications
    for med in meds_found:
        # We only check if we actually have mapping data for this med
        # to avoid penalizing correctly extracted unknown meds.
        # So we do a partial match against our keys:
        matched_key = next((k for k in MED_TO_CONDITION_MAP.keys() if k in med), None)
        
        if matched_key:
            expected_problems = MED_TO_CONDITION_MAP[matched_key]
            
            # Check if ANY expected problem (or a substring) exists in the problems_found set
            has_indication = False
            for ep in expected_problems:
                if any(ep in p for p in problems_found):
                    has_indication = True
                    break
                    
            if not has_indication:
                coherence_warnings.append({
                    "medication": med,
                    "missing_indication_context": expected_problems
                })
                
    return {
        "coherence_omission_warnings": coherence_warnings,
        "coherence_score": 1.0 - (len(coherence_warnings) / len(meds_found)) if meds_found else 1.0
    }
