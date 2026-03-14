import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def score_file_anomalies(data: list) -> dict:
    """
    Computes statistical anomalies for an entire file (list of entities).
    Returns file-level scores like duplication rate and anomaly score.
    """
    if not data:
        return {'anomaly_score': 0.0, 'duplicate_rate': 0.0}
        
    df = pd.DataFrame(data)
    
    # 1. Duplication Rate
    # Entities with exact same text, type, and heading
    dup_cols = ['entity', 'entity_type', 'heading']
    if all(c in df.columns for c in dup_cols):
        duplicates = df.duplicated(subset=dup_cols).sum()
        dup_rate = duplicates / len(df)
    else:
        dup_rate = 0.0
        
    # 2. File-level Anomaly Scoring via Isolation Forest features
    # (We build a tiny feature vector for this file to see if it's statistically weird 
    # compared to a normal distribution, though ideally we train this across *all* files.
    # For a per-file eval, we just return basic stats that the global reporter can use.)
    
    # Calculate some basic densities
    type_counts = df.get('entity_type', pd.Series()).value_counts(normalize=True).to_dict()
    assert_counts = df.get('assertion', pd.Series()).value_counts(normalize=True).to_dict()
    
    return {
        'duplicate_rate': float(dup_rate),
        'pct_medicine': float(type_counts.get('MEDICINE', 0.0)),
        'pct_procedure': float(type_counts.get('PROCEDURE', 0.0)),
        'pct_positive': float(assert_counts.get('POSITIVE', 0.0)),
        'total_entities': len(df)
    }

def calculate_tfidf_grounding(entity: str, context_text: str) -> float:
    """
    Calculate how well the entity is grounded in the source text using TF-IDF cosine similarity.
    Very low scores indicate the entity text might be hallucinated or heavily transformed.
    """
    if not entity or not context_text:
        return 0.0
        
    try:
        tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        tfidf_matrix = tfidf.fit_transform([entity, context_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(sim[0][0])
    except:
        return 0.0  # Fallback on parsing errors

def evaluate_entity_stats(entity_data: dict) -> dict:
    """
    Statistical per-entity checks.
    """
    entity = entity_data.get('entity', '')
    text = entity_data.get('text', '')
    
    # Check text grounding
    grounding_score = calculate_tfidf_grounding(entity, text)
    
    # An entity is poorly grounded if its character n-grams have <10% overlap with the source text
    is_ungrounded = grounding_score < 0.10 if len(entity) > 3 else False
    
    return {
        'grounding_score': grounding_score,
        'is_ungrounded_error': is_ungrounded
    }

def process_batch_anomalies(all_file_stats: list) -> list:
    """
    Given stats for ALL files, run Isolation Forest to find anomalous charts.
    """
    if len(all_file_stats) < 5:
        return all_file_stats  # Not enough data for isolation forest
        
    df = pd.DataFrame(all_file_stats)
    features = ['duplicate_rate', 'pct_medicine', 'pct_procedure', 'pct_positive', 'total_entities']
    
    X = df[features].fillna(0)
    
    # Fit Isolation Forest
    clf = IsolationForest(random_state=42, contamination=0.1) # Expect 10% anomalies
    preds = clf.fit_predict(X)
    
    # Add anomaly flags back to the stats
    # preds is -1 for outliers, 1 for inliers
    for i, stat in enumerate(all_file_stats):
        stat['is_statistical_anomaly'] = bool(preds[i] == -1)
        
    return all_file_stats
