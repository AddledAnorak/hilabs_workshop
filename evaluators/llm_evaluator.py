import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = None
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        # base_url=os.getenv("OPENAI_BASE_URL"),
        timeout=15.0,  # Higher timeout for batch
        max_retries=1
    )

def evaluate_file_llm_batch(file_data: list, context_text: str) -> dict:
    """
    Sends ONE prompt containing condensed summaries of all entities, 
    and returns an array of indices defining which ones contain errors.
    """
    if not client or not file_data:
        return {}
        
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Limit context text size
    context_text = context_text[:3000] if len(context_text) > 3000 else context_text
    
    # Condense entities to minimize token usage
    # We only send up to 50 entities max to avoid context bloat
    sample_entities = file_data[:50] 
    entities_str = "\n".join([
        f"[{i}] '{ent.get('entity', '')}' | Type: {ent.get('entity_type', '')} | Assert: {ent.get('assertion', '')} | Temp: {ent.get('temporality', '')}"
        for i, ent in enumerate(sample_entities)
    ])
    
    prompt = f"""
    You are evaluating clinical NLP extractions from a medical chart. 
    Review the source text snippet and the extracted entities below.
    
    SOURCE TEXT (First 3000 chars):
    \"\"\"
    {context_text}
    \"\"\"
    
    EXTRACTED ENTITIES:
    {entities_str}
    
    Identify any BLATANT ERRORS in the extractions. An error is:
    1. type_error: The entity string is clearly NOT the stated Type (e.g. "patient" is not a PROCEDURE).
    2. assertion_error: The text clearly states the OPPOSITE assertion (e.g. marked POSITIVE but text says "denies").
    
    Output strictly in JSON format matching exactly this schema:
    {{
       "flagged_entities": [
          {{"index": <int matching the bracket number>, "error_type": "<type_error or assertion_error>", "reason": "<short string>"}}
       ]
    }}
    If no errors, return an empty list for flagged_entities.
    """
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a clinical NLP evaluator. Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            timeout=15.0
        )
        result = json.loads(response.choices[0].message.content)
        
        # Build mapping of index -> error dictionary
        llm_flags = {}
        for flag in result.get('flagged_entities', []):
            idx = flag.get('index')
            err_type = flag.get('error_type')
            
            if idx is not None and isinstance(idx, int):
                llm_flags[idx] = {
                    'llm_entity_type_error': err_type == 'type_error',
                    'llm_assertion_error': err_type == 'assertion_error',
                    'llm_reason': flag.get('reason', '')
                }
                
        return llm_flags
    except Exception as e:
        print(f"LLM Batch Error: {e}")
        return {}
