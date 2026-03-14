import json
import os
import re

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def has_negation_cues(text):
    text = text.lower()
    patterns = [
        r'\bno\b', r'\bnot\b', r'\bdenies\b', r'\bnegative for\b', r'\bwithout\b',
        r'\babsent\b', r'\bresolved\b'
    ]
    return any(re.search(p, text) for p in patterns)

def has_historical_cues(text):
    text = text.lower()
    patterns = [
        r'\bhistory of\b', r'\bpast\b', r'\bprevious\b', r'\bresolved\b',
        r'\bformer\b', r'\bago\b'
    ]
    return any(re.search(p, text) for p in patterns)

def has_upcoming_cues(text):
    text = text.lower()
    patterns = [
        r'\bscheduled\b', r'\bfollow-up in\b', r'\btomorrow\b', r'\bnext\b',
        r'\bfuture\b', r'\bupcoming\b', r'\bwill\b'
    ]
    return any(re.search(p, text) for p in patterns)

def has_family_cues(text, heading):
    combined = (text + " " + heading).lower()
    patterns = [
        r'\bfamily history\b', r'\bmother\b', r'\bfather\b', r'\bbrother\b',
        r'\bsister\b', r'\bsibling\b', r'\bdaughter\b', r'\bson\b', r'\buncle\b', r'\baunt\b'
    ]
    return any(re.search(p, combined) for p in patterns)
