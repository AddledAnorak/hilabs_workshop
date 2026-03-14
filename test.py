import argparse
import os
import glob
from tqdm import tqdm

from evaluators.utils import load_json, save_json
from evaluators.rule_evaluator import evaluate_entity
from evaluators.statistical_evaluator import evaluate_entity_stats, score_file_anomalies, process_batch_anomalies
from evaluators.llm_evaluator import evaluate_file_llm_batch

# Only import embedding evaluator if requested to save on memory/load time
def get_embedding_evaluator():
    try:
        from evaluators.embedding_evaluator import evaluate_entity_semantics
        return evaluate_entity_semantics
    except ImportError:
        return None

def evaluate_file(input_path, output_path, use_llm=False, use_embeddings=True):
    data = load_json(input_path)
    
    total_entities = len(data)
    if total_entities == 0:
        print(f"Warning: {input_path} has no entities.")
        return None
        
    # Trackers for errors per dimension
    errors = {
        'entity_type': {'MEDICINE': 0, 'PROBLEM': 0, 'PROCEDURE': 0, 'TEST': 0, 
                        'VITAL_NAME': 0, 'IMMUNIZATION': 0, 'MEDICAL_DEVICE': 0, 
                        'MENTAL_STATUS': 0, 'SDOH': 0, 'SOCIAL_HISTORY': 0},
        'assertion': {'POSITIVE': 0, 'NEGATIVE': 0, 'UNCERTAIN': 0},
        'temporality': {'CURRENT': 0, 'CLINICAL_HISTORY': 0, 'UPCOMING': 0, 'UNCERTAIN': 0},
        'subject': {'PATIENT': 0, 'FAMILY_MEMBER': 0}
    }
    
    totals = {
        'entity_type': {k: 0 for k in errors['entity_type']},
        'assertion': {k: 0 for k in errors['assertion']},
        'temporality': {k: 0 for k in errors['temporality']},
        'subject': {k: 0 for k in errors['subject']}
    }
    
    date_accuracy_scores = []
    attribute_completeness_scores = []
    
    # Run batched LLM evaluation for the whole file at once (limit context and entities)
    llm_flags = {}
    if use_llm:
        # Get up to 3000 chars of source text from the first entity that has it
        context_text = next((e.get('text', '') for e in data if 'text' in e), "")
        llm_flags = evaluate_file_llm_batch(data, context_text)
        
    embedding_evaluator = get_embedding_evaluator() if use_embeddings else None
    
    entity_details_log = []

    for i, entity in enumerate(data):
        ent_type = entity.get('entity_type', '')
        assertion = entity.get('assertion', '')
        temporality = entity.get('temporality', '')
        subject = entity.get('subject', '')
        
        # Count totals
        if ent_type in totals['entity_type']: totals['entity_type'][ent_type] += 1
        if assertion in totals['assertion']: totals['assertion'][assertion] += 1
        if temporality in totals['temporality']: totals['temporality'][temporality] += 1
        if subject in totals['subject']: totals['subject'][subject] += 1
            
        # 1. Rule-based evaluation
        rule_errors = evaluate_entity(entity)
        
        # 2. Statistical / Grounding Evaluation
        stat_errors = evaluate_entity_stats(entity)
        
        # 3. Embedding (Semantic) Evaluation
        emb_errors = {}
        if embedding_evaluator:
            emb_errors = embedding_evaluator(entity.get('entity', ''), ent_type)
            
        # 4. LLM Evaluation (from precomputed batch)
        llm_err = llm_flags.get(i, {})
        
        # --- Weighted Ensemble Logic ---
        # An error is flagged if: 
        # (A) Rules explicitly catch it (highest confidence)
        # OR (B) Embeddings strong mismatch AND Stats ungrounded
        # OR (C) LLM explicitly flags it
        
        final_errors = {
            'entity_type_error': bool(
                rule_errors.get('entity_type_error') or 
                llm_err.get('llm_entity_type_error') or
                (emb_errors.get('semantic_type_error') and stat_errors.get('is_ungrounded_error'))
            ),
            'assertion_error': bool(
                rule_errors.get('assertion_error') or
                llm_err.get('llm_assertion_error')
            ),
            'temporality_error': rule_errors.get('temporality_error', False),
            'subject_error': rule_errors.get('subject_error', False)
        }
        
        # Keep entity inspection log for dashboard
        if any(final_errors.values()):
            entity_details_log.append({
                'entity': entity.get('entity', ''),
                'type': ent_type,
                'assertion': assertion,
                'errors': {k: v for k, v in final_errors.items() if v},
                'diagnostic_scores': {
                    'grounding': stat_errors.get('grounding_score', 0),
                    'semantic_confidence': emb_errors.get('semantic_confidence', 0)
                }
            })
                
        # Aggregate errors based on final ensemble
        if final_errors['entity_type_error'] and ent_type in errors['entity_type']:
            errors['entity_type'][ent_type] += 1
        if final_errors['assertion_error'] and assertion in errors['assertion']:
            errors['assertion'][assertion] += 1
        if final_errors['temporality_error'] and temporality in errors['temporality']:
            errors['temporality'][temporality] += 1
        if final_errors['subject_error'] and subject in errors['subject']:
            errors['subject'][subject] += 1
            
        if 'event_date_accuracy' in rule_errors:
            date_accuracy_scores.append(rule_errors['event_date_accuracy'])
        if 'attribute_completeness' in rule_errors:
            attribute_completeness_scores.append(rule_errors['attribute_completeness'])

    # Calculate final rates for JSON
    def calc_rate(err_counts, tot_counts):
        return {k: (err_counts[k] / tot_counts[k] if tot_counts[k] > 0 else 0.0) for k in err_counts}

    # File-level Statistical Anomalies
    file_stats = score_file_anomalies(data)
    
    report = {
        "file_name": os.path.basename(input_path),
        "entity_type_error_rate": calc_rate(errors['entity_type'], totals['entity_type']),
        "assertion_error_rate": calc_rate(errors['assertion'], totals['assertion']),
        "temporality_error_rate": calc_rate(errors['temporality'], totals['temporality']),
        "subject_error_rate": calc_rate(errors['subject'], totals['subject']),
        "event_date_accuracy": sum(date_accuracy_scores) / len(date_accuracy_scores) if date_accuracy_scores else 1.0,
        "attribute_completeness": sum(attribute_completeness_scores) / len(attribute_completeness_scores) if attribute_completeness_scores else 1.0,
        "file_statistics": file_stats,
        "error_details": entity_details_log
    }
    
    save_json(report, output_path)
    return report

def main():
    parser = argparse.ArgumentParser(description="Advanced Clinical AI Evaluator (V2)")
    parser.add_argument("input", nargs="?", help="Input JSON file")
    parser.add_argument("output", nargs="?", help="Output JSON file")
    parser.add_argument("--batch", action="store_true", help="Process all files in test_data directory")
    parser.add_argument("--llm", action="store_true", help="Use LLM for deeper evaluation")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable semantic embedding checks (faster)")
    
    args = parser.parse_args()
    
    if args.batch:
        input_dir = "test_data"
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        json_files = glob.glob(os.path.join(input_dir, "*", "*.json"))
        print(f"Found {len(json_files)} files. Starting V2 batch evaluation...")
        
        # Clear output dir
        for f in glob.glob(os.path.join(output_dir, "*.json")):
            os.remove(f)
            
        all_reports = []
        for filepath in tqdm(json_files):
            filename = os.path.basename(filepath)
            out_path = os.path.join(output_dir, filename)
            rep = evaluate_file(filepath, out_path, use_llm=args.llm, use_embeddings=not args.no_embeddings)
            if rep:
                all_reports.append(rep)
                
        # Run Isolation Forest on the combined file statistics
        flat_stats = [r['file_statistics'] for r in all_reports]
        flat_stats = process_batch_anomalies(flat_stats)
        
        # Inject the statistical anomaly flag back into the JSONs
        for idx, filepath in enumerate(json_files):
            filename = os.path.basename(filepath)
            out_path = os.path.join(output_dir, filename)
            if os.path.exists(out_path):
                data = load_json(out_path)
                data['file_statistics']['is_statistical_anomaly'] = flat_stats[idx].get('is_statistical_anomaly', False)
                save_json(data, out_path)
            
        print(f"Batch processing complete. Results saved in {output_dir}/")
        
    elif args.input and args.output:
        evaluate_file(args.input, args.output, use_llm=args.llm, use_embeddings=not args.no_embeddings)
        print(f"Evaluated {args.input} -> {args.output}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
