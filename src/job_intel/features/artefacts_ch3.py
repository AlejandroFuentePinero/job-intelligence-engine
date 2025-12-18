# src/job_intel/features/artefacts_ch3.py

"""
Chapter 3 artefact loaders.

This module centralises loading of all persisted artefacts required for
individual positioning (Chapter 3), ensuring consistent sources and schemas
across suitability, gap analysis, and competitiveness components.

Artefacts loaded here are treated as read-only.
"""

import pandas as pd

from src.job_intel.config import CH2_PROCESSED_DF, SKILL_PROB_MATRIX


def load_ch3_artefacts():
    """
    Load core Chapter 3 artefacts.

    Returns
    -------
    jobs_df : pd.DataFrame
        Modelling-ready jobs table produced at the end of Chapter 2.
        Must include identifiers, filters (state, sector, titles),
        and skill representations (PCs and/or skill flags).

    skill_prob_matrix : pd.DataFrame
        Skill probability matrix from Chapter 1.
        Must include `job_id` and one column per skill named `{skill}_prob`.

    Notes
    -----
    This function performs no transformation beyond loading.
    Validation of required columns is handled by downstream modules.
    """

    jobs_df = pd.read_csv(CH2_PROCESSED_DF)
    skill_prob_matrix = pd.read_csv(SKILL_PROB_MATRIX)

    return jobs_df, skill_prob_matrix
