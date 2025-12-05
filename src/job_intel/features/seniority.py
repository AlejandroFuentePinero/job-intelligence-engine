# src/job_intel/features/seniority.py

from __future__ import annotations

import re

# ---------------------------------------------------------------------
# 1. CANONICAL SENIORITY PATTERN LISTS (regex with word boundaries)
# ---------------------------------------------------------------------

principal = [
    r"\bprincipal\b",
]

executive = [
    r"\bchief\b",
    r"\bchief [a-z]+ officer\b",
    r"\bvice president\b",
    r"\bsvp\b",
    r"\bvp\b",
    r"\bexecutive\b",
]

assistant = [
    r"\bassistant\b",
]

supervisor = [
    r"\bsupervisor\b",
    r"\bsupervisory\b",
]

lead = [
    r"\bteam lead\b",
    r"\btech lead\b",
    r"\blead\b",  # safe: exact word, avoids 'leading', 'leadership'
    r"\bcoordinator\b",
]

senior = [
    r"\bsenior\b",
    r"\bsr\b",
    r"\bsr\.\b",
]

manager = [
    r"\bmanager\b",
    r"\bdirector\b",
    r"\bhead of\b",
    r"\bhead[- ]?[\w ]*department\b",
    r"\bofficer\b",
    r"\bavp\b",
    # We intentionally DO NOT add raw "\bhead\b" or "\bvice\b"
    # to avoid massive over-trigger from "ahead", "advice", etc.
]

mid = [
    r"\bmid[- ]?level\b",
    r"\bassociate\b",
    r"\bintermediate\b",
    r"\bmid\b",
    r"\bmid level\b",
]

junior = [
    r"\bjunior\b",
    r"\bjr\b",
    r"\bjr\.\b",
    r"\bentry level\b",
    r"\binternship\b",
    r"\bintern\b",
    r"\btrainee\b",
    r"\bunpaid\b",
    r"\bgraduate\b",
    r"\bgrad\b",
    r"\bentry\b",
    r"\bundergraduate\b",
]

# Priority order: higher first, as in the notebook
seniority_patterns = [
    ("principal", principal),
    ("executive", executive),
    ("manager", manager),
    ("lead", lead),
    ("senior", senior),
    ("mid", mid),
    ("junior", junior),
    ("assistant", assistant),
    ("supervisor", supervisor),
]


# ---------------------------------------------------------------------
# 2. CANONICAL REGEX-BASED DETECTOR (used after Roman numerals)
# ---------------------------------------------------------------------


def detect_seniority(text: str) -> str:
    """
    Detect seniority from free text using regex patterns with word boundaries.

    Uses the first 300 characters as a snippet (to reduce noise in long
    descriptions) and applies patterns in priority order:
        principal > executive > manager > lead > senior > mid > junior
        > assistant > supervisor

    Returns a label or 'unknown' if nothing matches.
    """
    if not isinstance(text, str):
        return "unknown"

    x = text.lower()
    snippet = x[:300]

    for label, patterns in seniority_patterns:
        for pat in patterns:
            if re.search(pat, snippet):
                return label

    return "unknown"


# ---------------------------------------------------------------------
# 3. TITLE / DESCRIPTION ENTRYPOINTS
# ---------------------------------------------------------------------


def detect_seniority_from_title(text: str) -> str:
    """
    Seniority detector for job titles.

    NOTE: Roman numerals are already handled upstream in clean_job_title()
    as 'seniority_roman'. This function is used only when there is no
    Roman numeral seniority, to pick up words like 'senior', 'manager', etc.
    """
    return detect_seniority(text)


def detect_seniority_from_description(text: str) -> str:
    """
    Seniority detector for job descriptions.

    For now, this uses the same regex-based detector. If we later decide to
    only use very strong cues (intern, trainee, etc.), we can restrict that
    logic here without changing the rest of the pipeline.
    """
    return detect_seniority(text)


# ---------------------------------------------------------------------
# 4. COMBINATION LOGIC
# ---------------------------------------------------------------------


def combine_seniority(desc_val: str, title_val: str) -> str:
    """
    Combine description-based and title-based seniority.

    DESCRIPTION-FIRST logic (matches the benchmark):
      - If description gives a label (not 'unknown'), use that.
      - Else, if title gives a label, use that.
      - Else 'unknown'.
    """
    if desc_val != "unknown":
        return desc_val
    elif title_val != "unknown":
        return title_val
    else:
        return "unknown"
