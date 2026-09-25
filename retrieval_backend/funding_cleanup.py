"""Conservative funding cleanup; raw graph records are retained for review."""
import ast
import json
import re


def clean_funding_name(name):
    """Return (names, reason). Never infer new funding from free text."""
    name = str(name or '').strip()
    lowered = name.casefold()
    if (re.search(r'\bnsf\s+grant\s+12345\b', lowered)
            or any(s in lowered for s in ('end with }', 'response format example',
                'so we output exactly', "i'll produce:", 'funding agency and grant number'))):
        return [], 'extraction_template'
    if lowered in ('', '.', 'none', 'null', 'n/a', 'not applicable', '[]', '{}'):
        return [], 'empty_placeholder'
    if name.startswith(('{', '[')):
        try:
            try:
                value = json.loads(name)
            except ValueError:
                value = ast.literal_eval(name)
        except (ValueError, SyntaxError):
            return [name], 'unparsed_preserved'
        if isinstance(value, dict) and set(value) == {'funding'}:
            value = value['funding']
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            result = []
            for item in value:
                cleaned, _ = clean_funding_name(item)
                result.extend(cleaned)
            return list(dict.fromkeys(result)), 'structured_list'
    return [name], 'unchanged'
