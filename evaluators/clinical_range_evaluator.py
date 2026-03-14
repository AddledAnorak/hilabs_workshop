def evaluate_vital_range(entity_data: dict) -> dict:
    """
    Validates if a VITAL_NAME or TEST numeric value falls within 
    clinically plausible bounds. If not, flags as a likely OCR error.
    """
    entity = entity_data.get('entity', '').lower()
    entity_type = entity_data.get('entity_type', '')
    metadata = entity_data.get('metadata_from_qa', {})
    
    if entity_type not in ["VITAL_NAME", "TEST"]:
        return {}
        
    value_relations = []
    
    # Extract the numeric value
    for rel in metadata.get('relations', []):
        if rel.get('entity_type') in ['TEST_VALUE', 'VITAL_NAME_VALUE']:
            value_relations.append(rel.get('entity', ''))
            
    if not value_relations:
        return {}
        
    # Standardize value extraction
    try:
        # Just grab the first number found in the value string
        import re
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", value_relations[0])
        if not nums:
            return {}
        val = float(nums[0])
    except ValueError:
        return {} # Can't parse, so we can't range-check
        
    errors = {}
    
    # Heart Rate
    if any(k in entity for k in ['heart rate', 'hr', 'pulse']):
        if val < 20 or val > 250:
            errors['clinical_range_error'] = True
            
    # Blood Pressure (Systolic - simplistic given we just grab the first number)
    elif 'blood pressure' in entity or 'bp' in entity:
        if val < 50 or val > 300:
            errors['clinical_range_error'] = True
            
    # Temperature (F or C combined)
    elif 'temp' in entity:
        if (val < 90 and val > 45) or (val > 110) or (val < 32):  
            # 90-110F or 32-45C are acceptable
            errors['clinical_range_error'] = True
            
    # Oxygen Sat
    elif any(k in entity for k in ['o2', 'spo2', 'saturation']):
        if val < 50 or val > 100:  # 100 is max, below 50 is essentially deceased
            errors['clinical_range_error'] = True
            
    # Glucose
    elif 'glucose' in entity or 'sugar' in entity:
        if val < 20 or val > 1000:
            errors['clinical_range_error'] = True
            
    return errors
