from collections import defaultdict

def build_cross_file_consensus(all_files_data: list):
    """
    Pass 1: Collects all entities across the entire batch to establish the
    most common 'entity_type' assignment for every unique entity string.
    """
    entity_counts = defaultdict(lambda: defaultdict(int))
    
    for file_entities in all_files_data:
        for ent in file_entities:
            # We use lowercased entity text as the key
            text_key = ent.get('entity', '').lower().strip()
            ent_type = ent.get('entity_type', '')
            if text_key and ent_type:
                entity_counts[text_key][ent_type] += 1
                
    consensus_map = {}
    for text_key, type_distribution in entity_counts.items():
        total_occurrences = sum(type_distribution.values())
        # If the entity appears >= 3 times total
        if total_occurrences >= 3:
            # Find the most common type
            dominant_type = max(type_distribution.items(), key=lambda x: x[1])
            dom_type_name, dom_count = dominant_type
            
            # If the dominant type is > 80% consensus, we save it as truth
            if dom_count / total_occurrences >= 0.8:
                consensus_map[text_key] = dom_type_name
                
    return consensus_map

def evaluate_cross_file_consistency(entity_data: dict, consensus_map: dict) -> dict:
    """
    Pass 2: For a single entity, verify its assigned type matches the global consensus.
    """
    if not consensus_map:
        return {}
        
    entity = entity_data.get('entity', '').lower().strip()
    assigned_type = entity_data.get('entity_type', '')
    
    if entity in consensus_map:
        true_type = consensus_map[entity]
        if assigned_type != true_type:
            return {
                'cross_file_mismatch_error': True,
                'consensus_type': true_type
            }
            
    return {}
