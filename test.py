import argparse
import os
import glob
from tqdm import tqdm

from evaluators.utils import load_json, save_json
from evaluators.rule_evaluator import evaluate_entity
from evaluators.statistical_evaluator import evaluate_entity_stats, score_file_anomalies, process_batch_anomalies
from evaluators.llm_evaluator import evaluate_file_llm_batch
from evaluators.embedding_evaluator import evaluate_entity_semantics_offline
from evaluators.vocabulary_evaluator import evaluate_entity_vocabulary
from evaluators.cross_file_evaluator import build_cross_file_consensus, evaluate_cross_file_consistency
from evaluators.clinical_range_evaluator import evaluate_vital_range
from evaluators.coherence_evaluator import evaluate_chart_coherence

def evaluate_file(data, input_filename, output_path, use_llm=False, consensus_map=None):
    total_entities = len(data)
    if total_entities == 0:
        return None
        
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
    
    llm_flags = {}
    if use_llm:
        context_text = next((e.get('text', '') for e in data if 'text' in e), "")
        llm_flags = evaluate_file_llm_batch(data, context_text)
        
    entity_details_log = []

    for i, entity in enumerate(data):
        ent_type = entity.get('entity_type', '')
        assertion = entity.get('assertion', '')
        temporality = entity.get('temporality', '')
        subject = entity.get('subject', '')
        
        if ent_type in totals['entity_type']: totals['entity_type'][ent_type] += 1
        if assertion in totals['assertion']: totals['assertion'][assertion] += 1
        if temporality in totals['temporality']: totals['temporality'][temporality] += 1
        if subject in totals['subject']: totals['subject'][subject] += 1
            
        # V3 Ensemble Evaluation Suite
        rule_errors = evaluate_entity(entity)
        stat_errors = evaluate_entity_stats(entity)
        emb_errors = evaluate_entity_semantics_offline(entity.get('entity', ''), ent_type)
        vocab_errors = evaluate_entity_vocabulary(entity)
        cross_errors = evaluate_cross_file_consistency(entity, consensus_map) if consensus_map else {}
        range_errors = evaluate_vital_range(entity)
        llm_err = llm_flags.get(i, {})
        
        # Merge semantic signals
        strong_type_mismatch = bool(
            vocab_errors.get('vocabulary_mismatch_error') or 
            cross_errors.get('cross_file_mismatch_error') or
            (emb_errors.get('errors', {}).get('semantic_mismatch_error') and stat_errors.get('is_ungrounded_error'))
        )
        
        final_errors = {
            'entity_type_error': bool(
                rule_errors.get('entity_type_error') or 
                llm_err.get('llm_entity_type_error') or
                strong_type_mismatch
            ),
            'assertion_error': bool(
                rule_errors.get('assertion_error') or
                llm_err.get('llm_assertion_error')
            ),
            'temporality_error': rule_errors.get('temporality_error', False),
            'subject_error': rule_errors.get('subject_error', False),
            'span_alignment_error': rule_errors.get('span_alignment_error', False),
            'clinical_range_error': range_errors.get('clinical_range_error', False)
        }
        
        # Logging for Dashboard
        if any(final_errors.values()):
            entity_details_log.append({
                'entity': entity.get('entity', ''),
                'type': ent_type,
                'assertion': assertion,
                'errors': {k: v for k, v in final_errors.items() if v},
                'diagnostic_scores': {
                    'grounding': stat_errors.get('grounding_score', 0),
                    'semantic_confidence': emb_errors.get('diagnostic_scores', {}).get('semantic_confidence', 0)
                },
                'vocab_correction': vocab_errors.get('vocabulary_correction'),
                'consensus_correction': cross_errors.get('consensus_type')
            })
                
        # Tally metrics
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

    coherence_stats = evaluate_chart_coherence(data)
    file_stats = score_file_anomalies(data)
    
    def calc_rate(err_counts, tot_counts):
        return {k: (err_counts[k] / tot_counts[k] if tot_counts[k] > 0 else 0.0) for k in err_counts}

    report = {
        "file_name": input_filename,
        "entity_type_error_rate": calc_rate(errors['entity_type'], totals['entity_type']),
        "assertion_error_rate": calc_rate(errors['assertion'], totals['assertion']),
        "temporality_error_rate": calc_rate(errors['temporality'], totals['temporality']),
        "subject_error_rate": calc_rate(errors['subject'], totals['subject']),
        "event_date_accuracy": sum(date_accuracy_scores) / len(date_accuracy_scores) if date_accuracy_scores else 1.0,
        "attribute_completeness": sum(attribute_completeness_scores) / len(attribute_completeness_scores) if attribute_completeness_scores else 1.0,
        "file_statistics": file_stats,
        "coherence_statistics": coherence_stats,
        "error_details": entity_details_log
    }
    
    # Create simplified report matching exactly what the user requested
    simple_report = {
        "file_name": input_filename,
        "entity_type_error_rate": report["entity_type_error_rate"],
        "assertion_error_rate": report["assertion_error_rate"],
        "temporality_error_rate": report["temporality_error_rate"],
        "subject_error_rate": report["subject_error_rate"],
        "event_date_accuracy": report["event_date_accuracy"],
        "attribute_completeness": report["attribute_completeness"]
    }
    
    if output_path:
        # Save simplified version to the main output path
        save_json(simple_report, output_path)
        
        # Save full detailed version for the dashboard/report generator
        detailed_dir = os.path.join(os.path.dirname(output_path), "detailed")
        os.makedirs(detailed_dir, exist_ok=True)
        save_json(report, os.path.join(detailed_dir, input_filename))
        
    return report

def main():
    parser = argparse.ArgumentParser(description="Advanced Clinical AI Evaluator (V3)")
    parser.add_argument("--batch", action="store_true", help="Process all files in test_data directory")
    parser.add_argument("--llm", action="store_true", help="Use LLM for deeper evaluation")
    
    args = parser.parse_args()
    
    if args.batch:
        input_dir = "test_data"
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        json_files = glob.glob(os.path.join(input_dir, "*", "*.json"))
        print(f"Found {len(json_files)} files. Starting V3 Multi-Pass Batch Evaluation...")
        
        # Clear output dir
        for f in glob.glob(os.path.join(output_dir, "*.json")):
            os.remove(f)
            
        # PASS 1: Load all data into memory for global analysis
        print("Pass 1: Building global knowledge maps...")
        all_data = []
        for filepath in json_files:
            all_data.append(load_json(filepath))
            
        consensus_map = build_cross_file_consensus(all_data)
        print(f"-> Established strong consensus for {len(consensus_map)} unique clinical entities.")
            
        # PASS 2: Evaluate individual files
        print("\nPass 2: Evaluating individual charts...")
        all_reports = []
        for idx, filepath in enumerate(tqdm(json_files)):
            filename = os.path.basename(filepath)
            out_path = os.path.join(output_dir, filename)
            
            rep = evaluate_file(all_data[idx], filename, out_path, use_llm=args.llm, consensus_map=consensus_map)
            if rep:
                all_reports.append(rep)
                
        # PASS 3: File-level aggregations
        print("\nPass 3: Statistical Anomalies...")
        flat_stats = [r['file_statistics'] for r in all_reports]
        flat_stats = process_batch_anomalies(flat_stats)
        
        # Re-inject Statistical Anomaly flag into BOTH the simple and detailed JSONs
        for idx, filepath in enumerate(json_files):
            filename = os.path.basename(filepath)
            
            # Update detailed JSON
            detailed_path = os.path.join(output_dir, "detailed", filename)
            if os.path.exists(detailed_path):
                data = load_json(detailed_path)
                data['file_statistics']['is_statistical_anomaly'] = flat_stats[idx].get('is_statistical_anomaly', False)
                save_json(data, detailed_path)
            
            # Simple JSON doesn't have file_statistics, so we skip injecting it there 
            # as per the user's preferred format.
            
        print(f"\nV3 Batch processing complete. Results saved in {output_dir}/")
        
    else:
        print("Please use --batch to run the evaluation.")
        parser.print_help()

if __name__ == "__main__":
    main()
