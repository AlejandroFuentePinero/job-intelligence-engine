import re
from . import skills_taxonomy

# Build a mapping: (domain, level) -> list_of_tokens
TAXA_LEVEL_LISTS = {}

for list_name, (domain, level) in skills_taxonomy.LIST_META.items():
    tokens = getattr(skills_taxonomy, list_name)
    TAXA_LEVEL_LISTS.setdefault((domain, level), tokens)

"""
The output will look something like this:

    This is the master lookup; the foundation of the extractor

{
    ("core_programming", "basic"): ["python", "java", "sql", ...],
    ("core_programming", "intermediate"): ["javascript", "c++", ...],
    ("machine_learning", "intermediate"): ["pytorch", "tensorflow", ...],
    ...
}

"""

ALLOWED_SHORT_TOKENS = {"r"}  # maybe "c" later if we decide


def token_in_text(token: str, text: str) -> bool:
    text = text.lower()
    token = token.lower().strip()

    if " " in token:
        pattern = rf"\b{re.escape(token)}\b"
        return re.search(pattern, text) is not None

    if len(token) <= 2 and token not in ALLOWED_SHORT_TOKENS:
        return False

    pattern = rf"\b{re.escape(token)}\b"
    return re.search(pattern, text) is not None


"""

This function decides whether the word in the diccionary exist in the
text we are extracting skills from

"""


def extract_domain_level_flags(text: str) -> dict:
    """
    Given a raw job description/title text, return a dict of
    { "<domain>__<level>": 0/1 } indicating whether any skill
    from that (domain, level) bucket was found.

    This is the v1, dictionary-based extractor.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text_low = text.lower()

    # Initialise features: all zeros
    features = {f"{domain}__{level}": 0 for (domain, level) in TAXA_LEVEL_LISTS.keys()}

    # For each (domain, level), see if any of its tokens appears
    for (domain, level), tokens in TAXA_LEVEL_LISTS.items():
        feature_name = f"{domain}__{level}"

        for tok in tokens:
            if token_in_text(tok, text_low):
                features[feature_name] = 1
                break  # stop at first hit for this bucket

    return features


"""
This loops over the map above.
    - Searches whether the token in each domain, level exist in the text
    - If True, then we add a 1 to the feature dictionary. This will become the multi hot key later

"""


def explain_matches(text: str):
    text_low = text.lower()
    hits = {}

    for (domain, level), tokens in TAXA_LEVEL_LISTS.items():
        bucket = f"{domain}__{level}"
        hits[bucket] = []

        for tok in tokens:
            if token_in_text(tok, text_low):
                hits[bucket].append(tok)

    return hits
