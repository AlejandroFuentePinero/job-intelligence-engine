# src/job_intel/features/text_cleaning.py

from __future__ import annotations

import re
import pandas as pd
from nltk.corpus import stopwords

# You already used this in the notebook; make sure the stopwords
# corpus is downloaded in your environment at least once:
#   >>> import nltk; nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
whitelist_short = {"c", "r", "js", "go", "c++", "c#", "s3", "ab"}


def surface_cleaning(text: str) -> str:
    """
    Surface-level cleaning of job description text.

    - lowercases
    - splits on most punctuation (keeps + and # inside tokens)
    - drops stopwords
    - filters out very short tokens (except a small whitelist)
    - removes heavily numeric / garbage tokens
    - keeps a de-duplicated, order-preserving sequence of tokens
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Split everything on punctuation except + and #
    text = re.sub(r"[^a-z0-9+#]+", " ", text)

    tokens = text.split()
    cleaned = []
    seen = set()

    for tok in tokens:
        tok = tok.strip(".,:;!?")

        # Skip slash-combined tokens e.g. analyst/sql, developer/analyst
        if "/" in tok:
            for part in tok.split("/"):
                if len(part) >= 2:
                    cleaned.append(part)
            continue

        # stopwords
        if tok in stop_words:
            continue

        # length filter except whitelist
        if len(tok) < 3 and tok not in whitelist_short:
            continue

        # digits / heavily numeric tokens
        if tok.isdigit() or len(re.findall(r"\d", tok)) >= 3:
            continue

        # alphanumeric garbage
        if re.match(r"^[a-z]+[0-9]+[a-z]+$", tok):
            continue

        # too many punctuation markers
        if sum(c in "+.#" for c in tok) > 2:
            continue

        if tok not in seen:
            cleaned.append(tok)
            seen.add(tok)

    return " ".join(cleaned)


def add_description_features(
    df: pd.DataFrame,
    *,
    raw_col: str = "Job Description",
    out_col: str = "job_description_clean",
) -> pd.DataFrame:
    """
    Add a cleaned description column to the dataframe using surface_cleaning.
    """
    df = df.copy()
    df[out_col] = df[raw_col].apply(surface_cleaning)
    return df
