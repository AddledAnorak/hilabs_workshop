import argparse
import os
import glob
from tqdm import tqdm

from evaluators.utils import load_json, save_json
from evaluators.rule_evaluator import evaluate_entity
from evaluators.llm_evaluator import evaluate_entity_llm

def evaluate_file(input_path, output_path, use_llm=False):
    data = load_json(input_path)
    # data is a list of entity dicts
    
    total_entities = len(data)
    if total_entities == 0:
        print(f"Warning: {input_path} has no entities.")
        return
        
    # Trackers for errors per dimension
    errors = {
        'entity_type': {'MEDICINE': 0, 'PROBLEM': 0, 'PROCEDURE': 0, 'TEST': 0, 
                        'VITAL_NAME': 0, 'IMMUNIZATION': 0, 'MEDICAL_DEVICE': 0, 
                        'MENTAL_STATUS': 0, 'SDOH': 0, 'SOCIAL_HISTORY': 0},
        'assertion': {'POSITIVE': 0, 'NEGATIVE': 0, 'UNCERTAIN': 0},
        'temporality': {'CURRENT': 0, 'CLINICAL_HISTORY': 0, 'UPCOMING': 0, 'UNCERTAIN': 0},
        'subject': {'PATIENT': 0, 'FAMILY_MEMBER': 0}
    }
    
    # Trackers for totals to calculate rates
    totals = {
        'entity_type': {k: 0 for k in errors['entity_type']},
        'assertion': {k: 0 for k in errors['assertion']},
        'temporality': {k: 0 for k in errors['temporality']},
        'subject': {k: 0 for k in errors['subject']}
    }
    
    date_accuracy_scores = []
    attribute_completeness_scores = []

    for entity in data:
        ent_type = entity.get('entity_type', '')
        assertion = entity.get('assertion', '')
        temporality = entity.get('temporality', '')
        subject = entity.get('subject', '')
        
        # Count totals
        if ent_type in totals['entity_type']: totals['entity_type'][ent_type] += 1
        if assertion in totals['assertion']: totals['assertion'][assertion] += 1
        if temporality in totals['temporality']: totals['temporality'][temporality] += 1
        if subject in totals['subject']: totals['subject'][subject] += 1
            
        # Run rule-based evaluator
        rule_errors = evaluate_entity(entity)
        
        # Run LLM evaluator if requested
        if use_llm:
            llm_errors = evaluate_entity_llm(entity)
            if llm_errors.get('llm_entity_type_error'):
                rule_errors['entity_type_error'] = True
                
        # Aggregate errors
        if rule_errors.get('entity_type_error') and ent_type in errors['entity_type']:
            errors['entity_type'][ent_type] += 1
            
        if rule_errors.get('assertion_error') and assertion in errors['assertion']:
            errors['assertion'][assertion] += 1
            
        if rule_errors.get('temporality_error') and temporality in errors['temporality']:
            errors['temporality'][temporality] += 1
            
        if rule_errors.get('subject_error') and subject in errors['subject']:
            errors['subject'][subject] += 1
            
        if 'event_date_accuracy' in rule_errors:
            date_accuracy_scores.append(rule_errors['event_date_accuracy'])
            
        if 'attribute_completeness' in rule_errors:
            attribute_completeness_scores.append(rule_errors['attribute_completeness'])

    # Calculate rates
    def calc_rate(err_counts, tot_counts):
        return {k: (err_counts[k] / tot_counts[k] if tot_counts[k] > 0 else 0.0) for k in err_counts}

    report = {
        "file_name": os.path.basename(input_path),
        "entity_type_error_rate": calc_rate(errors['entity_type'], totals['entity_type']),
        "assertion_error_rate": calc_rate(errors['assertion'], totals['assertion']),
        "temporality_error_rate": calc_rate(errors['temporality'], totals['temporality']),
        "subject_error_rate": calc_rate(errors['subject'], totals['subject']),
        "event_date_accuracy": sum(date_accuracy_scores) / len(date_accuracy_scores) if date_accuracy_scores else 1.0,
        "attribute_completeness": sum(attribute_completeness_scores) / len(attribute_completeness_scores) if attribute_completeness_scores else 1.0
    }
    
    save_json(report, output_path)

def main():
    parser = argparse.ArgumentParser(description="Clinical AI Pipeline Evaluator")
    parser.add_argument("input", nargs="?", help="Input JSON file or directory")
    parser.add_argument("output", nargs="?", help="Output JSON file or directory")
    parser.add_argument("--batch", action="store_true", help="Process all files in test_data directory")
    parser.add_argument("--llm", action="store_true", help="Use LLM for deeper evaluation")
    
    args = parser.parse_args()
    
    # Process batch mode
    if args.batch:
        input_dir = "test_data"
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all JSONs in test_data subdirectories
        json_files = glob.glob(os.path.join(input_dir, "*", "*.json"))
        
        print(f"Found {len(json_files)} files. Starting batch evaluation...")
        for filepath in tqdm(json_files):
            filename = os.path.basename(filepath)
            out_path = os.path.join(output_dir, filename)
            evaluate_file(filepath, out_path, use_llm=args.llm)
            
        print(f"Batch processing complete. Results saved in {output_dir}/")
        
    # Process single file
    elif args.input and args.output:
        evaluate_file(args.input, args.output, use_llm=args.llm)
        print(f"Evaluated {args.input} -> {args.output}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
