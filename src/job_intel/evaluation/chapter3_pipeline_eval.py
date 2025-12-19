# src/job_intel/evaluation/chapter3_pipeline_eval.py

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from src.job_intel.positioning import run_positioning
from src.job_intel.features.artefacts_ch3 import load_ch3_artefacts
from src.job_intel.features.skills_pca import SKILL_COLS
from src.job_intel.features.candidate_competitiveness import add_competitiveness
from src.job_intel.features.candidate_selection import candidate_set_construction


# -----------------------------
# Helpers
# -----------------------------
_TOL = 1e-9


def _load_or_raise(
    df: Optional[pd.DataFrame] = None, skill: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return artefacts for Chapter 3.

    If df or skill_prob_matrix are not provided, loads them from disk via load_ch3_artefacts().

    Returns
    -------
    (df, skill_prob_matrix)
    """
    if df is None or skill is None:
        df_loaded, skill_loaded = load_ch3_artefacts()
        return df_loaded, skill_loaded
    return df, skill


def _assert_or_raise(cond: bool, msg: str) -> None:
    """Raise AssertionError if cond is False."""
    if not cond:
        raise AssertionError(msg)


def _align_by_job_id(
    left: pd.DataFrame,
    right: pd.DataFrame,
    cols_left: List[str],
    cols_right: List[str],
    job_id_col: str = "job_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two dataframes on the intersection of job_id values and return aligned views.

    Notes
    -----
    This avoids pandas "identically-labeled Series" issues and makes comparisons robust to ordering.
    """
    l = left[[job_id_col] + cols_left].set_index(job_id_col)
    r = right[[job_id_col] + cols_right].set_index(job_id_col)
    common = l.index.intersection(r.index)
    return l.loc[common], r.loc[common]


# -----------------------------
# Tests
# -----------------------------


def test_filtered_artefacts(
    df: Optional[pd.DataFrame] = None, skill: Optional[pd.DataFrame] = None
) -> None:
    """
    Validate that Chapter 3 artefacts are correctly filtered and aligned.

    Checks
    ------
    1) No job row has zero skills across SKILL_COLS
    2) job_id universe matches between df and skill_prob_matrix (set equality)

    Raises
    ------
    AssertionError if any check fails.
    """
    df, skill = _load_or_raise(df, skill)

    print("TEST - Job filter TEST")
    print(
        "Check whether the filter for jobs with zero skills was implemented correctly."
    )
    print("---------------------------")

    zero_skill_jobs = int((df[SKILL_COLS].sum(axis=1) == 0).sum())
    _assert_or_raise(
        zero_skill_jobs == 0,
        f"Filter FAIL: found {zero_skill_jobs} jobs with zero skills.",
    )
    print("✅ Filter applied correctly to artefacts: PASS")

    df_ids = set(df["job_id"])
    skill_ids = set(skill["job_id"])
    if df_ids != skill_ids:
        missing_in_skill = list(df_ids - skill_ids)[:5]
        missing_in_df = list(skill_ids - df_ids)[:5]
        raise AssertionError(
            "Alignment FAIL: df and skill_prob_matrix have different job_id universes. "
            f"Example missing_in_skill={missing_in_skill}, missing_in_df={missing_in_df}"
        )
    print("✅ df and skill matrix job_id universe matches: PASS")


def test_invariance(
    random_job: Optional[int] = 1034,
    df: Optional[pd.DataFrame] = None,
    run_sensitivity: bool = True,
    current_state: str = "ALL",
    job_title_family: str = "data_scientist",
    job_title_rich: Optional[str] = "ML_AI_data_scientist",
    target_sectors: Optional[List[str]] = None,
    salary_target: int = 250000,
    explain_skills: bool = False,
    top_k_gaps: int = 50,
    return_top_n_jobs: int = 200,
) -> Tuple[
    Any,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
    Any,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:
    """
    Determinism / invariance test.

    Runs run_positioning() twice with identical inputs and asserts:
    - No unexpected NAs in key outputs
    - Identical suitability / competitiveness / gaps (within rounding tolerance)
    - Identical job_id ordering (positional equality)
    - If sensitivity enabled: identical sensitivity tables (within rounding tolerance)

    Parameters
    ----------
    random_job
        job_id used to source a realistic skill_text from df["Job Description"].
        If not present, a deterministic fallback is used (first job_id).
    df
        Optional jobs dataframe. If None, loads artefacts and uses that df.

    Returns
    -------
    (profile_r1, jobs_df_r1, gaps_df_r1, sensitivity_r1, profile_r2, jobs_df_r2, gaps_df_r2, sensitivity_r2)

    Raises
    ------
    ValueError for NA issues; AssertionError for invariance failures.
    """
    df, _ = _load_or_raise(df=df, skill=None)

    # Choose a valid job_id for skill_text extraction
    if random_job is None or random_job not in set(df["job_id"]):
        random_job = int(df["job_id"].iloc[0])

    _assert_or_raise(
        "Job Description" in df.columns,
        "df must contain a 'Job Description' column to build skill_text for invariance test.",
    )

    skill_text = df.loc[df["job_id"] == random_job, "Job Description"].iloc[0]

    profile_r1, jobs_df_r1, gaps_df_r1, sensitivity_r1 = run_positioning(
        skill_text=skill_text,
        current_state=current_state,
        job_title_family=job_title_family,
        job_title_rich=job_title_rich,
        target_sectors=target_sectors,
        salary_target=salary_target,
        explain_skills=explain_skills,
        top_k_gaps=top_k_gaps,
        return_top_n_jobs=return_top_n_jobs,
        run_sensitivity=run_sensitivity,
    )

    profile_r2, jobs_df_r2, gaps_df_r2, sensitivity_r2 = run_positioning(
        skill_text=skill_text,
        current_state=current_state,
        job_title_family=job_title_family,
        job_title_rich=job_title_rich,
        target_sectors=target_sectors,
        salary_target=salary_target,
        explain_skills=explain_skills,
        top_k_gaps=top_k_gaps,
        return_top_n_jobs=return_top_n_jobs,
        run_sensitivity=run_sensitivity,
    )

    print("TEST - INVARIANCE TEST")
    print(
        "Run the pipeline twice with the same arguments and check for identical expected values."
    )
    print("---------------------------")

    # NA checks (hard failure)
    if (
        jobs_df_r1[["suitability", "competitiveness_index"]].isna().any().any()
        or jobs_df_r2[["suitability", "competitiveness_index"]].isna().any().any()
    ):
        raise ValueError("dfs have unexpected NAs in suitability/competitiveness_index")
    print("✅ NA test: PASS")

    # Positional equality on job_id (ordering stability)
    same_order = (
        jobs_df_r1["job_id"]
        .reset_index(drop=True)
        .equals(jobs_df_r2["job_id"].reset_index(drop=True))
    )
    _assert_or_raise(
        same_order,
        "Positional equality FAIL: job ordering differs between identical runs.",
    )
    print("✅ Positional equality: PASS")

    # Robust comparisons aligned by job_id
    j1, j2 = _align_by_job_id(
        jobs_df_r1,
        jobs_df_r2,
        cols_left=["suitability", "competitiveness_index"],
        cols_right=["suitability", "competitiveness_index"],
    )
    g1, g2 = _align_by_job_id(
        (
            gaps_df_r1.rename(columns={"skill": "job_id"}).copy()
            if "job_id" not in gaps_df_r1.columns
            else gaps_df_r1
        ),
        (
            gaps_df_r2.rename(columns={"skill": "job_id"}).copy()
            if "job_id" not in gaps_df_r2.columns
            else gaps_df_r2
        ),
        cols_left=["skill_gap"],
        cols_right=["skill_gap"],
    )

    # Suitability invariance
    suit_equal = (j1["suitability"].round(5) == j2["suitability"].round(5)).mean() == 1
    _assert_or_raise(
        suit_equal,
        f"Suitability invariance FAIL: max |Δ|={(j1['suitability']-j2['suitability']).abs().max()}",
    )
    print("✅ Identical suitability scores: PASS")

    # Competitiveness invariance
    comp_equal = (
        j1["competitiveness_index"].round(5) == j2["competitiveness_index"].round(5)
    ).mean() == 1
    _assert_or_raise(
        comp_equal,
        f"Competitiveness invariance FAIL: max |Δ|={(j1['competitiveness_index']-j2['competitiveness_index']).abs().max()}",
    )
    print("✅ Identical competitiveness scores: PASS")

    # Skill gap invariance (if gaps keyed by 'skill', the alignment helper above may be imperfect;
    # for now, compare by row order after rounding, as your notebook checks did)
    if "skill" in gaps_df_r1.columns and "skill" in gaps_df_r2.columns:
        gg1 = (
            gaps_df_r1.sort_values("skill").reset_index(drop=True)["skill_gap"].round(5)
        )
        gg2 = (
            gaps_df_r2.sort_values("skill").reset_index(drop=True)["skill_gap"].round(5)
        )
        gap_equal = (gg1 == gg2).mean() == 1
        _assert_or_raise(
            gap_equal, f"Skill gap invariance FAIL: max |Δ|={(gg1-gg2).abs().max()}"
        )
    else:
        gap_equal = (g1["skill_gap"].round(5) == g2["skill_gap"].round(5)).mean() == 1
        _assert_or_raise(gap_equal, "Skill gap invariance FAIL (alignment-based).")
    print("✅ Identical skill gap: PASS")

    # Shape invariance
    _assert_or_raise(
        jobs_df_r1.shape == jobs_df_r2.shape,
        "Shape FAIL: jobs_df shape differs between identical runs.",
    )
    print("✅ Identical shape: PASS")

    # Sensitivity invariance
    if run_sensitivity:
        s1 = sensitivity_r1["suitability"]["spearman_rho_vs_baseline"].round(5)
        s2 = sensitivity_r2["suitability"]["spearman_rho_vs_baseline"].round(5)
        _assert_or_raise(
            (s1 == s2).mean() == 1,
            f"Sensitivity(suitability) FAIL: max |Δ|={(s1-s2).abs().max()}",
        )
        print("✅ Identical suitability sensitivity: PASS")

        c1 = sensitivity_r1["competitiveness"]["spearman_rho_vs_baseline"].round(5)
        c2 = sensitivity_r2["competitiveness"]["spearman_rho_vs_baseline"].round(5)
        _assert_or_raise(
            (c1 == c2).mean() == 1,
            f"Sensitivity(competitiveness) FAIL: max |Δ|={(c1-c2).abs().max()}",
        )
        print("✅ Identical competitiveness sensitivity: PASS")

    return (
        profile_r1,
        jobs_df_r1,
        gaps_df_r1,
        sensitivity_r1,
        profile_r2,
        jobs_df_r2,
        gaps_df_r2,
        sensitivity_r2,
    )


def test_induce_zero_jobs_error(
    current_state: str = "TX",
    job_title_family: str = "data_scientist",
    job_title_rich: str = "ML_AI_data_scientist",
    target_sectors: Optional[List[str]] = None,
) -> None:
    """
    Candidate selection boundary test: filters yield zero jobs => explicit ValueError.

    PASS if run_positioning raises ValueError with the expected failure mode.
    FAIL if no error is raised.
    """
    if target_sectors is None:
        target_sectors = ["Real Estate"]

    print("TEST - ZERO JOBS ERROR")
    print("---------------------------")

    try:
        _ = run_positioning(
            skill_text="python",
            current_state=current_state,
            job_title_family=job_title_family,
            job_title_rich=job_title_rich,
            target_sectors=target_sectors,
            salary_target=250000,
            explain_skills=False,
            top_k_gaps=50,
            return_top_n_jobs=200,
            run_sensitivity=True,
        )
    except ValueError as e:
        print(f"✅ Zero-jobs constraint raises ValueError as expected: PASS ({e})")
        return

    raise AssertionError(
        "Expected ValueError for zero-job constraints, but pipeline did not raise."
    )


def test_one_row_return(
    return_top_n_jobs: int = 1,
) -> Tuple[Any, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Single-job candidate-set boundary test.

    PASS if pipeline completes and returns a jobs_df with <= return_top_n_jobs rows.
    """
    profile_1j, jobs_df_1j, gaps_df_1j, sensitivity_1j = run_positioning(
        skill_text="python",
        current_state="ALL",
        job_title_family="data_scientist",
        job_title_rich="ML_AI_data_scientist",
        target_sectors=None,
        salary_target=250000,
        explain_skills=False,
        top_k_gaps=1,
        return_top_n_jobs=return_top_n_jobs,
        run_sensitivity=True,
    )

    _assert_or_raise(
        len(jobs_df_1j) <= return_top_n_jobs,
        "Single-row return FAIL: too many jobs returned.",
    )
    print("✅ Single-job return: PASS")
    return profile_1j, jobs_df_1j, gaps_df_1j, sensitivity_1j


def test_force_skill_col_mismatch(
    df: Optional[pd.DataFrame] = None,
    skill: Optional[pd.DataFrame] = None,
    missing_prob_col: str = "productivity_workflow__intermediate_prob",
) -> None:
    """
    Artefact integrity test: missing required {skill}_prob column should raise KeyError.

    PASS if add_competitiveness raises KeyError.
    """
    df, skill = _load_or_raise(df, skill)

    profile, candidates_df = candidate_set_construction(
        df=df,
        skill_text="python",
        current_state="ALL",
        job_title_family="data_scientist",
        job_title_rich=None,
        target_sectors=None,
        salary_target=250000,
        explain_skills=False,
    )

    print("TEST - SKILL COL MISMATCH")
    print("---------------------------")

    bad_skill = skill.drop(columns=[missing_prob_col], errors="ignore").copy()

    try:
        _ = add_competitiveness(profile, candidates_df, bad_skill, use_rarity=True)
    except KeyError as e:
        print(f"✅ Missing prob column raises KeyError as expected: PASS ({e})")
        return

    raise AssertionError(
        "Expected KeyError for missing prob column, but add_competitiveness did not raise."
    )


def test_force_skill_row_mismatch(
    df: Optional[pd.DataFrame] = None,
    skill: Optional[pd.DataFrame] = None,
    nrows_to_clip: int = 5,
) -> None:
    """
    Artefact integrity test: job_id mismatch between candidates_df and skill_prob_matrix should raise KeyError.

    Implementation:
    - Build candidates_df from real df
    - Remove probabilities for a few candidate job_ids from skill_prob_matrix
    - Expect KeyError in add_competitiveness

    PASS if KeyError is raised.
    """
    df, skill = _load_or_raise(df, skill)

    profile, candidates_df = candidate_set_construction(
        df=df,
        skill_text="python",
        current_state="ALL",
        job_title_family="data_scientist",
        job_title_rich=None,
        target_sectors=None,
        salary_target=250000,
        explain_skills=False,
    )

    print("TEST - SKILL ROW MISMATCH")
    print("---------------------------")

    bad_ids = candidates_df["job_id"].head(nrows_to_clip)
    skill_prob_mismatch = skill[~skill["job_id"].isin(bad_ids)].copy()

    try:
        _ = add_competitiveness(
            profile, candidates_df, skill_prob_mismatch, use_rarity=True
        )
    except KeyError as e:
        print(f"✅ job_id mismatch raises KeyError as expected: PASS ({e})")
        return

    raise AssertionError(
        "Expected KeyError for job_id mismatch, but add_competitiveness did not raise."
    )


def test_suitability_behaviour_weights() -> None:
    """
    Behaviour sanity: increasing w_skill should benefit high-skill-match jobs more than low-skill-match jobs.

    This is not strict monotonicity per job; it is a directional sanity check (group-level).
    PASS criterion: mean Δ suitability for high-skill-match group > mean Δ suitability for low-skill-match group.
    """
    _, jobs_df_r1, _, _ = run_positioning(
        skill_text="python, sql, cloud",
        current_state="ALL",
        job_title_family="data_scientist",
        job_title_rich="ML_AI_data_scientist",
        target_sectors=None,
        w_skill=0.7,
        salary_target=250000,
        explain_skills=False,
        top_k_gaps=50,
        return_top_n_jobs=200,
        run_sensitivity=False,
    )

    _, jobs_df_r2, _, _ = run_positioning(
        skill_text="python, sql, cloud",
        current_state="ALL",
        job_title_family="data_scientist",
        job_title_rich="ML_AI_data_scientist",
        target_sectors=None,
        w_skill=0.9,
        salary_target=250000,
        explain_skills=False,
        top_k_gaps=50,
        return_top_n_jobs=200,
        run_sensitivity=False,
    )

    d1 = jobs_df_r1[["job_id", "suitability", "skill_match_norm"]].set_index("job_id")
    d2 = jobs_df_r2[["job_id", "suitability"]].set_index("job_id")

    common = d1.index.intersection(d2.index)
    d1 = d1.loc[common]
    d2 = d2.loc[common]

    delta = d2["suitability"] - d1["suitability"]
    median = d1["skill_match_norm"].median()

    hi = float(delta[d1["skill_match_norm"] >= median].mean())
    lo = float(delta[d1["skill_match_norm"] < median].mean())

    print("TEST - SUITABILITY WEIGHT BEHAVIOUR")
    print("---------------------------")
    print("mean Δ (high skill_match):", hi)
    print("mean Δ (low  skill_match):", lo)

    _assert_or_raise(
        hi > lo,
        "Suitability weight behaviour FAIL: high-skill-match group did not benefit more.",
    )
    print("✅ Suitability weight behaviour: PASS")


def test_suitability_behaviour_salary() -> None:
    """
    Behaviour sanity: increasing salary_target should not increase suitability (per job), when aligned by job_id.

    PASS if there are zero violations: suitability(high target) > suitability(low target).
    """
    _, jobs_df_r1, _, _ = run_positioning(
        skill_text="python, sql, cloud",
        current_state="ALL",
        job_title_family="data_scientist",
        job_title_rich="ML_AI_data_scientist",
        target_sectors=None,
        w_skill=0.7,
        salary_target=250000,
        explain_skills=False,
        top_k_gaps=50,
        return_top_n_jobs=200,
        run_sensitivity=False,
    )

    _, jobs_df_r2, _, _ = run_positioning(
        skill_text="python, sql, cloud",
        current_state="ALL",
        job_title_family="data_scientist",
        job_title_rich="ML_AI_data_scientist",
        target_sectors=None,
        w_skill=0.7,
        salary_target=550000,
        explain_skills=False,
        top_k_gaps=50,
        return_top_n_jobs=200,
        run_sensitivity=False,
    )

    a = jobs_df_r1.set_index("job_id")["suitability"]
    b = jobs_df_r2.set_index("job_id")["suitability"]

    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]

    violations = int((b > a + _TOL).sum())

    print("TEST - SUITABILITY SALARY BEHAVIOUR")
    print("---------------------------")
    print("violations:", violations, "out of", len(common))

    _assert_or_raise(
        violations == 0, f"Suitability salary behaviour FAIL: {violations} violations."
    )
    print("✅ Suitability salary behaviour: PASS")


def test_competitiveness_behaviour_skill(
    df: Optional[pd.DataFrame] = None, skill: Optional[pd.DataFrame] = None
) -> None:
    """
    Behaviour sanity: adding skills should not increase competitiveness (fixed candidate set),
    removing skills should not decrease competitiveness (fixed candidate set).

    Notes
    -----
    This test holds candidates_df fixed (from base profile) and recomputes competitiveness
    by swapping the user profile only.

    PASS if:
    - For all jobs: comp_add <= comp_base (within tolerance)
    - For all jobs: comp_remove >= comp_base (within tolerance)
    """
    _, skill = _load_or_raise(df=None, skill=skill)

    profile_base, candidates_df_base, _, _ = run_positioning(
        skill_text="python, sql, coaching, cloud",
        current_state="ALL",
        job_title_family="data_scientist",
        job_title_rich="ML_AI_data_scientist",
        target_sectors=None,
        w_skill=0.7,
        salary_target=250000,
        explain_skills=False,
        top_k_gaps=50,
        return_top_n_jobs=200,
        run_sensitivity=False,
    )

    profile_remove, _, _, _ = run_positioning(
        skill_text="python, cloud",
        current_state="ALL",
        job_title_family="data_scientist",
        job_title_rich="ML_AI_data_scientist",
        target_sectors=None,
        w_skill=0.7,
        salary_target=250000,
        explain_skills=False,
        top_k_gaps=50,
        return_top_n_jobs=200,
        run_sensitivity=False,
    )

    profile_add, _, _, _ = run_positioning(
        skill_text="python, sql, coaching, cloud, git, gis",
        current_state="ALL",
        job_title_family="data_scientist",
        job_title_rich="ML_AI_data_scientist",
        target_sectors=None,
        w_skill=0.7,
        salary_target=250000,
        explain_skills=False,
        top_k_gaps=50,
        return_top_n_jobs=200,
        run_sensitivity=False,
    )

    comp_base = add_competitiveness(profile_base, candidates_df_base, skill)
    comp_remove = add_competitiveness(profile_remove, candidates_df_base, skill)
    comp_add = add_competitiveness(profile_add, candidates_df_base, skill)

    print("TEST - COMPETITIVENESS SKILL BEHAVIOUR")
    print("---------------------------")

    _assert_or_raise(
        (comp_add["job_id"] == comp_base["job_id"]).mean() == 1,
        "Index alignment FAIL (add).",
    )
    _assert_or_raise(
        (comp_remove["job_id"] == comp_base["job_id"]).mean() == 1,
        "Index alignment FAIL (remove).",
    )
    print("✅ Identical job_id alignment: PASS")

    add_ok = (
        comp_add["competitiveness_index"] <= comp_base["competitiveness_index"] + _TOL
    ).mean() == 1
    rem_ok = (
        comp_remove["competitiveness_index"]
        >= comp_base["competitiveness_index"] - _TOL
    ).mean() == 1
    _assert_or_raise(add_ok, "Competitive skill behaviour add FAIL.")
    _assert_or_raise(rem_ok, "Competitive skill behaviour remove FAIL.")
    print("✅ Competitiveness monotonicity (add/remove): PASS")

    add_m_ok = (
        comp_add["expected_missing_norm"] <= comp_base["expected_missing_norm"] + _TOL
    ).mean() == 1
    rem_m_ok = (
        comp_remove["expected_missing_norm"]
        >= comp_base["expected_missing_norm"] - _TOL
    ).mean() == 1
    _assert_or_raise(add_m_ok, "Missing skill behaviour add FAIL.")
    _assert_or_raise(rem_m_ok, "Missing skill behaviour remove FAIL.")
    print("✅ Expected-missing monotonicity (add/remove): PASS")


def test_sensitivity() -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Sensitivity module sanity checks.

    Checks
    ------
    - weight sums: w_skill + w_salary == 1 (row-wise, tolerance)
    - baseline exists: (0.7, 0.3) has spearman_rho_vs_baseline == 1
    - rho bounds: within [-1, 1]
    - non-degenerate: at least one non-baseline rho < 1
    - ordering (weak): w_skill is monotone increasing

    Returns
    -------
    (sensitivity_dict, S_df, C_df)
    """
    _, _, _, sensitivity = run_positioning(
        skill_text="python, sql, cloud",
        current_state="ALL",
        job_title_family="data_scientist",
        job_title_rich="ML_AI_data_scientist",
        target_sectors=None,
        w_skill=0.7,
        salary_target=250000,
        explain_skills=False,
        top_k_gaps=50,
        return_top_n_jobs=200,
        run_sensitivity=True,
    )

    S = sensitivity["suitability"].copy()
    C = sensitivity["competitiveness"].copy()

    print("TEST - SENSITIVITY")
    print("---------------------------")
    print("Suitability table:\n", S, "\n")
    print("Competitiveness table:\n", C, "\n")

    _assert_or_raise(
        ((S["w_skill"] + S["w_salary"] - 1).abs().max() < _TOL),
        "Suitability weights FAIL: not normalised.",
    )
    _assert_or_raise(
        ((C["w_skill"] + C["w_salary"] - 1).abs().max() < _TOL),
        "Competitiveness weights FAIL: not normalised.",
    )
    print("✅ Weight normalisation: PASS")

    base_mask_S = (S["w_skill"].sub(0.7).abs() < _TOL) & (
        S["w_salary"].sub(0.3).abs() < _TOL
    )
    base_mask_C = (C["w_skill"].sub(0.7).abs() < _TOL) & (
        C["w_salary"].sub(0.3).abs() < _TOL
    )

    _assert_or_raise(
        base_mask_S.sum() == 1,
        "Suitability baseline FAIL: baseline row missing or not unique.",
    )
    _assert_or_raise(
        base_mask_C.sum() == 1,
        "Competitiveness baseline FAIL: baseline row missing or not unique.",
    )

    _assert_or_raise(
        float(S.loc[base_mask_S, "spearman_rho_vs_baseline"].iloc[0]) == 1.0,
        "Suitability baseline FAIL: rho != 1 at baseline.",
    )
    _assert_or_raise(
        float(C.loc[base_mask_C, "spearman_rho_vs_baseline"].iloc[0]) == 1.0,
        "Competitiveness baseline FAIL: rho != 1 at baseline.",
    )
    print("✅ Baseline exists + rho==1: PASS")

    _assert_or_raise(
        S["spearman_rho_vs_baseline"].between(-1, 1).all(),
        "Suitability rho bounds FAIL.",
    )
    _assert_or_raise(
        C["spearman_rho_vs_baseline"].between(-1, 1).all(),
        "Competitiveness rho bounds FAIL.",
    )
    print("✅ Spearman bounds [-1,1]: PASS")

    nondeg_S = (S.loc[~base_mask_S, "spearman_rho_vs_baseline"] < 1 - _TOL).any()
    nondeg_C = (C.loc[~base_mask_C, "spearman_rho_vs_baseline"] < 1 - _TOL).any()
    _assert_or_raise(
        nondeg_S, "Suitability degenerate FAIL: all non-baseline rhos are 1."
    )
    _assert_or_raise(
        nondeg_C, "Competitiveness degenerate FAIL: all non-baseline rhos are 1."
    )
    print("✅ Non-degenerate sensitivity: PASS")

    _assert_or_raise(
        S["w_skill"].is_monotonic_increasing,
        "Suitability ordering FAIL: w_skill not monotone increasing.",
    )
    _assert_or_raise(
        C["w_skill"].is_monotonic_increasing,
        "Competitiveness ordering FAIL: w_skill not monotone increasing.",
    )
    print("✅ Deterministic ordering (w_skill monotone): PASS")

    _assert_or_raise(
        S[["w_skill", "w_salary"]].duplicated().sum() == 0,
        "Suitability duplicates FAIL: repeated weight configs.",
    )
    _assert_or_raise(
        C[["w_skill", "w_salary"]].duplicated().sum() == 0,
        "Competitiveness duplicates FAIL: repeated weight configs.",
    )
    print("✅ One row per config: PASS")

    _assert_or_raise(
        not S.isna().any().any(), "Suitability shape FAIL: contains missing values."
    )
    _assert_or_raise(
        not C.isna().any().any(), "Competitiveness shape FAIL: contains missing values."
    )
    print("✅ No missing values: PASS")

    return sensitivity, S, C


def smoke_test_ch3(jobs_df: pd.DataFrame, gaps_df: pd.DataFrame) -> None:
    """
    End-to-end smoke test for Chapter 3 outputs.

    Checks
    ------
    - jobs_df and gaps_df exist and are non-empty
    - required columns exist
    - key score columns are within [0, 1]
    - jobs sorted by suitability descending
    - gap sanity: user-owned skills must have zero gap; gaps in [0, 1]

    Raises
    ------
    AssertionError on any failure.
    """
    print("TEST - CH3 SMOKE")
    print("---------------------------")

    _assert_or_raise(isinstance(jobs_df, pd.DataFrame), "jobs_df is not a DataFrame.")
    _assert_or_raise(isinstance(gaps_df, pd.DataFrame), "gaps_df is not a DataFrame.")
    _assert_or_raise(len(jobs_df) > 0, "No jobs returned.")
    _assert_or_raise(len(gaps_df) > 0, "No skill gaps returned.")

    required_job_cols = [
        "job_id",
        "suitability",
        "skill_match_norm",
        "salary_score",
        "competitiveness_index",
    ]
    missing_job_cols = [c for c in required_job_cols if c not in jobs_df.columns]
    _assert_or_raise(not missing_job_cols, f"Missing job columns: {missing_job_cols}")

    _assert_or_raise(
        jobs_df["suitability"].between(0, 1).all(), "Suitability outside [0,1]."
    )
    _assert_or_raise(
        jobs_df["skill_match_norm"].between(0, 1).all(),
        "Skill match norm outside [0,1].",
    )
    _assert_or_raise(
        jobs_df["salary_score"].between(0, 1).all(), "Salary score outside [0,1]."
    )

    _assert_or_raise(
        jobs_df["suitability"].is_monotonic_decreasing,
        "Jobs not sorted by suitability desc.",
    )

    required_gap_cols = ["skill", "job_skill_rate", "user_skill", "skill_gap"]
    missing_gap_cols = [c for c in required_gap_cols if c not in gaps_df.columns]
    _assert_or_raise(not missing_gap_cols, f"Missing gap columns: {missing_gap_cols}")

    _assert_or_raise(
        (gaps_df.loc[gaps_df["user_skill"] == 1, "skill_gap"] == 0).all(),
        "User-owned skills should not have gaps.",
    )
    _assert_or_raise(
        gaps_df["skill_gap"].between(0, 1).all(), "Skill gaps outside [0,1]."
    )

    print("✅ Chapter 3 smoke tests passed")
