import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = None
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

def evaluate_entity_llm(entity_data: dict) -> dict:
    # If no LLM, return empty dict or default
    if not client:
        return {}
        
    entity = entity_data.get('entity', '')
    entity_type = entity_data.get('entity_type', '')
    text = entity_data.get('text', '')
    
    prompt = f"""
    Given the following clinical text and an extracted entity, evaluate if the extraction is correct.
    Text: "{text}"
    Extracted Entity: "{entity}"
    Assigned Type: "{entity_type}"
    
    Respond in strict JSON with the following boolean fields (true if there's an ERROR, false if CORRECT):
    - is_type_error: true if the type doesn't fit the entity or if it's an OCR artifact.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # or whatever model is configured, wait, fallback to gpt-3.5-turbo if needed
            messages=[
                {"role": "system", "content": "You are a clinical NLP evaluator. Output only strict JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return {
            'llm_entity_type_error': result.get('is_type_error', False)
        }
    except Exception as e:
        print(f"LLM Error: {e}")
        return {}
