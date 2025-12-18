# src/job_intel/features/userprofile_skill_processing.py

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import pandas as pd

from src.job_intel.features.skill_extractor import (
    extract_domain_level_flags,
    explain_matches,
)
from src.job_intel.features.skills_pca import transform_skills_to_pca, SKILL_COLS


def user_skill_processor(
    skill_text: str,
    explain_skills: bool = False,
) -> Dict[str, Any]:
    """
    Convert user-provided skill text into:
      1) canonical 27-skill binary vector (wide)
      2) PCA skill components (PC1..PCn)
      3) long-form extracted skill flags
      4) optional explanation of token→skill matches

    Parameters
    ----------
    skill_text:
        Free-form text (resume excerpt / skills list / summary).
    explain_skills:
        If True, include token-level explanations from `explain_matches`.

    Returns
    -------
    dict with keys:
      - "skill_flags_long": DataFrame [skill, pred_val]
      - "skill_vector_wide": DataFrame 1×27 ordered by SKILL_COLS
      - "skill_pcs": DataFrame [skill_PC1..]
      - "skill_explanations": dict or None
    """
    # --- Input validation (fail fast) ---
    if not isinstance(skill_text, str):
        raise TypeError(f"skill_text must be a str, got {type(skill_text)}")
    skill_text = skill_text.strip()

    # Allow empty text (means all-zero skill vector)
    extracted: Dict[str, int] = extract_domain_level_flags(skill_text)

    # --- Long-form flags (debuggable / mergeable) ---
    skill_flags_long = pd.DataFrame(
        {"skill": list(extracted.keys()), "pred_val": list(extracted.values())}
    )

    # --- Wide vector in canonical order (PCA + models depend on this) ---
    skill_vector = pd.DataFrame(
        [{k: extracted.get(k, 0) for k in SKILL_COLS}],
        columns=SKILL_COLS,
    )

    # --- PCA projection ---
    skill_pcs = transform_skills_to_pca(skill_vector)

    # --- Optional explanations ---
    skill_explanations: Optional[dict] = None
    if explain_skills:
        # This can be expensive; compute only when requested.
        skill_explanations = explain_matches(skill_text)

    return {
        "skill_flags_long": skill_flags_long,
        "skill_vector": skill_vector,
        "skill_pcs": skill_pcs,
        "skill_explanations": skill_explanations,
    }
