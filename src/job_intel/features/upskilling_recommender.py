# src/job_intel/features/upskilling_recommender.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.job_intel.features.job_recommender import job_recommender
from src.job_intel.features.job_explanations import build_job_explanations
from src.job_intel.features.skill_extractor import explain_matches


# -----------------------------
# Helpers
# -----------------------------
def _require_keys(d: dict[str, Any], keys: list[str], *, where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"{where} is missing required keys: {missing}")


def _require_cols(df: pd.DataFrame, cols: list[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where} is missing required columns: {missing}")


def _dedup_tokens(tokens: list[Any], *, max_n: int) -> list[str]:
    """Case-insensitive dedup, keeps order, strips whitespace, drops empties."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        t = str(tok).strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= int(max_n):
            break
    return out


def _quantile_10(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    return float(np.quantile(x, 0.10))


def _mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    return float(x.mean())


def _coalesce_first(
    df: pd.DataFrame, candidates: list[str], out_col: str
) -> pd.DataFrame:
    """
    Create/overwrite out_col with the first available candidate column.
    Used to robustly recover a canonical column after merges that suffix columns.
    """
    for c in candidates:
        if c in df.columns:
            df[out_col] = df[c]
            return df
    return df


def _ensure_job_description(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a canonical 'Job Description' column.
    Handles common variants and merge suffixes.
    """
    if "Job Description" in df.columns:
        return df

    candidates = [
        "job_description",
        "description",
        "Job Description_base",
        "Job Description_x",
        "Job Description_y",
        "job_description_x",
        "job_description_y",
        "description_x",
        "description_y",
    ]
    df = _coalesce_first(df, candidates=candidates, out_col="Job Description")

    if "Job Description" not in df.columns:
        for c in df.columns:
            lc = str(c).lower()
            if (
                "job description" in lc
                or lc == "job_description"
                or lc == "description"
            ):
                df["Job Description"] = df[c]
                break

    if "Job Description" not in df.columns:
        df["Job Description"] = ""

    return df


def _get_skill_vector(rec: dict[str, Any]) -> pd.DataFrame | None:
    """
    Return profile['derived']['skill_vector'] as a single-row DataFrame if present, else None.
    """
    try:
        prof = rec["profile"]
        vec = prof["derived"]["skill_vector"]
        if isinstance(vec, pd.DataFrame) and len(vec) == 1:
            return vec
        return None
    except Exception:
        return None


@dataclass(frozen=True)
class UpskillRankingWeights:
    """
    Weights for the composite ranking score.
    """

    w_promote: float = 5.0
    w_stretch: float = 1.0
    w_best: float = 0.5
    w_demote: float = 10.0
    w_tail: float = 1.0


# -----------------------------
# Main
# -----------------------------
def upskill_recommender(
    *,
    skill_text: str = "",
    current_state: str = "ALL",
    job_title_family: str | None = None,
    job_title_rich: str | None = None,
    target_sectors: list[str] | None = None,
    salary_target: float | None = None,
    explain_skills: bool | None = None,
    # Explanation controls
    tau: float = 0.50,
    # Token injection controls (per family)
    n_tokens_per_family: int = 3,
    # Ranking controls
    weights: UpskillRankingWeights = UpskillRankingWeights(),
    demotion_tol: float = 0.0,  # strict guardrail by default
    top_n_skills: int = 3,
    # Pass-through to job_recommender (keep defaults consistent with Chapter 4)
    w_skill: float = 0.7,
    w_salary: float = 0.3,
    top_k_gaps: int = 200,
    return_top_n_jobs: int | None = 6200,
    run_sensitivity: bool = False,
    # Output / UX
    print_report: bool = True,
) -> dict[str, Any]:
    """
    Chapter 4 — Upskilling Recommender (v1)
    """
    if not (0.0 <= float(tau) <= 1.0):
        raise ValueError(f"tau must be in [0,1], got {tau}")
    if int(n_tokens_per_family) <= 0:
        raise ValueError("n_tokens_per_family must be >= 1")
    if int(top_n_skills) <= 0:
        raise ValueError("top_n_skills must be >= 1")

    skill_text_clean = str(skill_text or "").strip()
    skill_text_lc = skill_text_clean.lower()

    metric_cols = [
        "skill_match_norm",
        "suitability",
        "expected_missing_norm",
        "competitiveness_index",
        "bucket",
        "score",
    ]

    # Local wrapper to enforce frozen-universe rule:
    # positioning.py requires return_top_n_jobs=None when candidate_override_df is provided.
    def _run_rec(
        *, skill_text_in: str, candidate_override_df: pd.DataFrame | None
    ) -> dict[str, Any]:
        engine_return_top_n = (
            None if candidate_override_df is not None else return_top_n_jobs
        )
        return job_recommender(
            skill_text=skill_text_in,
            current_state=current_state,
            job_title_family=job_title_family,
            job_title_rich=job_title_rich,
            target_sectors=target_sectors,
            salary_target=salary_target,
            explain_skills=explain_skills,
            w_skill=w_skill,
            w_salary=w_salary,
            top_k_gaps=top_k_gaps,
            return_top_n_jobs=engine_return_top_n,
            run_sensitivity=run_sensitivity,
            candidate_override_df=candidate_override_df,
            verbose=False,
        )

    # -----------------------------
    # 1) Baseline run (includes scored_universe)
    # -----------------------------
    baseline_rec = _run_rec(skill_text_in=skill_text_clean, candidate_override_df=None)

    _require_keys(baseline_rec, ["tables"], where="baseline_rec")
    _require_keys(
        baseline_rec["tables"], ["scored_universe"], where="baseline_rec['tables']"
    )

    base_universe = baseline_rec["tables"]["scored_universe"].copy()
    base_universe = _ensure_job_description(base_universe)
    _require_cols(
        base_universe, ["job_id"] + metric_cols, where="baseline scored_universe"
    )

    # freeze universe by job_id only (minimal contract)
    override_df = base_universe[["job_id"]].copy()

    # baseline skill-vector (for no-op detection)
    baseline_vec = _get_skill_vector(baseline_rec)
    if baseline_vec is not None:
        baseline_vec = baseline_vec.reindex(sorted(baseline_vec.columns), axis=1)

    # -----------------------------
    # 2) Frozen-universe run + explanations (missing_families computed here)
    # -----------------------------
    universe_rec = _run_rec(
        skill_text_in=skill_text_clean, candidate_override_df=override_df
    )

    explanations = build_job_explanations(rec=universe_rec, tau=tau, validate=True)
    _require_keys(explanations, ["tables"], where="explanations")
    _require_keys(
        explanations["tables"],
        ["scored_universe_explained"],
        where="explanations['tables']",
    )

    explained_universe = explanations["tables"]["scored_universe_explained"].copy()
    explained_universe = _ensure_job_description(explained_universe)
    _require_cols(
        explained_universe,
        ["job_id", "bucket", "missing_families", "n_missing_families"],
        where="scored_universe_explained",
    )

    # -----------------------------
    # 3) Build missing_dict from *stretch* jobs (family -> tokens)
    # -----------------------------
    stretch = explained_universe.loc[explained_universe["bucket"] == "stretch"].copy()

    base_for_desc = base_universe[["job_id", "Job Description"]].copy()
    stretch = stretch.merge(
        base_for_desc,
        how="left",
        on="job_id",
        validate="many_to_one",
        suffixes=("", "_base"),
    )
    stretch = _ensure_job_description(stretch)

    stretch["matches_dict"] = (
        stretch["Job Description"].fillna("").apply(explain_matches)
    )

    stretch["set_missing_families"] = stretch["missing_families"].apply(
        lambda x: set(x) if isinstance(x, list) else set()
    )
    stretch["missing_matches_dict"] = [
        {fam: toks for fam, toks in md.items() if fam in miss}
        for md, miss in zip(stretch["matches_dict"], stretch["set_missing_families"])
    ]

    missing_dict: dict[str, list[str]] = {}
    for md in stretch["missing_matches_dict"].tolist():
        for fam, toks in md.items():
            missing_dict.setdefault(fam, []).extend(list(toks))

    expected_missing_fams = (
        set().union(*stretch["set_missing_families"].tolist())
        if len(stretch)
        else set()
    )

    if len(expected_missing_fams) == 0:
        raise ValueError(
            "No stretch jobs in the scored universe; cannot derive upskilling candidates."
        )
    if len(missing_dict) == 0:
        raise ValueError(
            "Derived stretch missing families but could not extract any matching tokens from descriptions. "
            "Check explain_matches() / Job Description availability."
        )

    # -----------------------------
    # 4) Baseline scenario table (for deltas)
    # -----------------------------
    baseline_jobs = base_universe[["job_id", "Job Description"] + metric_cols].copy()
    baseline_jobs["upskill_scenario"] = "baseline"

    baseline_map = baseline_jobs[
        [
            "job_id",
            "skill_match_norm",
            "suitability",
            "expected_missing_norm",
            "competitiveness_index",
            "bucket",
            "score",
        ]
    ].rename(
        columns={
            "skill_match_norm": "baseline_skill_match_norm",
            "suitability": "baseline_suitability",
            "expected_missing_norm": "baseline_expected_missing_norm",
            "competitiveness_index": "baseline_competitiveness_index",
            "bucket": "baseline_bucket",
            "score": "baseline_score",
        }
    )

    # -----------------------------
    # 5) Run upskill scenarios (frozen universe) + no-op checks
    # -----------------------------
    scenario_tables: list[pd.DataFrame] = [baseline_jobs]
    scenario_meta: list[dict[str, Any]] = []

    family_injection: dict[str, list[str]] = {}
    for fam, toks in missing_dict.items():
        cleaned = _dedup_tokens(toks, max_n=50)
        cleaned = [t for t in cleaned if t.lower() not in skill_text_lc]
        family_injection[fam] = cleaned[: int(n_tokens_per_family)]

    for fam, toks in family_injection.items():
        if not toks:
            scenario_meta.append(
                {
                    "family": fam,
                    "status": "skipped",
                    "reason": "no_tokens_after_filter",
                    "tokens": [],
                }
            )
            continue

        upskill_text = (skill_text_clean + " " + " ".join(toks)).strip()
        if upskill_text == skill_text_clean:
            scenario_meta.append(
                {
                    "family": fam,
                    "status": "skipped",
                    "reason": "text_no_change",
                    "tokens": toks,
                }
            )
            continue

        up_rec = _run_rec(skill_text_in=upskill_text, candidate_override_df=override_df)

        if baseline_vec is not None:
            up_vec = _get_skill_vector(up_rec)
            if up_vec is None:
                scenario_meta.append(
                    {
                        "family": fam,
                        "status": "skipped",
                        "reason": "missing_skill_vector",
                        "tokens": toks,
                    }
                )
                continue
            up_vec = up_vec.reindex(sorted(up_vec.columns), axis=1)
            if up_vec.equals(baseline_vec):
                scenario_meta.append(
                    {
                        "family": fam,
                        "status": "skipped",
                        "reason": "skill_vector_no_effect",
                        "tokens": toks,
                    }
                )
                continue

        _require_keys(up_rec, ["tables"], where=f"upskill rec ({fam})")
        _require_keys(
            up_rec["tables"], ["scored_universe"], where=f"upskill rec ({fam}) tables"
        )

        up_uni = up_rec["tables"]["scored_universe"].copy()
        up_uni = _ensure_job_description(up_uni)
        _require_cols(
            up_uni, ["job_id"] + metric_cols, where=f"upskill scored_universe ({fam})"
        )

        up_table = up_uni[["job_id", "Job Description"] + metric_cols].copy()
        up_table["upskill_scenario"] = f"upskill_{fam}"
        scenario_tables.append(up_table)

        scenario_meta.append(
            {"family": fam, "status": "kept", "reason": None, "tokens": toks}
        )

    upskill_rec_df = pd.concat(scenario_tables, axis=0, ignore_index=True)
    scenario_meta_df = pd.DataFrame(scenario_meta)

    counts = upskill_rec_df["upskill_scenario"].value_counts()
    baseline_n = int(counts.get("baseline", 0))
    bad = counts[counts != baseline_n]
    if len(bad) > 0:
        raise ValueError(
            "Upskilling scenario universe changed size (should be frozen). "
            f"Baseline n={baseline_n}. Mismatched scenarios: {bad.to_dict()}"
        )

    # -----------------------------
    # 6) Compute deltas vs baseline (per job, per scenario)
    # -----------------------------
    upskill_only = upskill_rec_df.loc[
        upskill_rec_df["upskill_scenario"] != "baseline"
    ].copy()
    upskill_df = upskill_only.merge(
        baseline_map, how="left", on="job_id", validate="many_to_one"
    )

    if upskill_df[["baseline_bucket", "baseline_score"]].isna().any().any():
        raise ValueError(
            "Baseline merge failed for some job_id rows; check job_id alignment."
        )

    upskill_df["delta_skill_match_norm"] = (
        upskill_df["skill_match_norm"] - upskill_df["baseline_skill_match_norm"]
    ) * 100
    upskill_df["delta_suitability"] = (
        upskill_df["suitability"] - upskill_df["baseline_suitability"]
    ) * 100
    upskill_df["delta_expected_missing_norm"] = (
        upskill_df["expected_missing_norm"]
        - upskill_df["baseline_expected_missing_norm"]
    ) * 100
    upskill_df["delta_competitiveness_index"] = (
        upskill_df["competitiveness_index"]
        - upskill_df["baseline_competitiveness_index"]
    ) * 100
    upskill_df["delta_score"] = (
        upskill_df["score"] - upskill_df["baseline_score"]
    ) * 100

    upskill_df["bucket_movement"] = np.where(
        (upskill_df["bucket"] == upskill_df["baseline_bucket"]),
        "unchanged",
        np.where(
            (upskill_df["bucket"] == "best_now")
            & (upskill_df["baseline_bucket"] == "stretch"),
            "promoted",
            "demoted",
        ),
    )

    # -----------------------------
    # 7) Scenario-level summary + composite ranking score
    # -----------------------------
    rows: list[dict[str, Any]] = []
    for scen, g in upskill_df.groupby("upskill_scenario", sort=False):
        g_best = g[g["baseline_bucket"] == "best_now"]
        g_stretch = g[g["baseline_bucket"] == "stretch"]

        n_best0 = int(len(g_best))
        n_stretch0 = int(len(g_stretch))

        n_promoted = int(
            ((g["baseline_bucket"] == "stretch") & (g["bucket"] == "best_now")).sum()
        )
        n_demoted = int(
            ((g["baseline_bucket"] == "best_now") & (g["bucket"] == "stretch")).sum()
        )

        promotion_rate = (n_promoted / n_stretch0) if n_stretch0 > 0 else float("nan")
        demotion_rate = (n_demoted / n_best0) if n_best0 > 0 else float("nan")

        rows.append(
            {
                "upskill_scenario": scen,
                "n_best0": n_best0,
                "n_stretch0": n_stretch0,
                "n_promoted": n_promoted,
                "promotion_rate": promotion_rate,
                "n_demoted": n_demoted,
                "demotion_rate": demotion_rate,
                "mean_delta_score_best": _mean(g_best["delta_score"]),
                "mean_delta_score_stretch": _mean(g_stretch["delta_score"]),
                "p10_delta_score_best": _quantile_10(g_best["delta_score"]),
                "mean_delta_comp_best": _mean(g_best["delta_competitiveness_index"]),
                "mean_delta_comp_stretch": _mean(
                    g_stretch["delta_competitiveness_index"]
                ),
            }
        )

    upskill_summary = pd.DataFrame(rows).set_index("upskill_scenario")

    tail_penalty = (
        upskill_summary["p10_delta_score_best"]
        .where(upskill_summary["p10_delta_score_best"] < 0, 0.0)
        .abs()
        .fillna(0.0)
    )

    upskill_summary["upskill_impact_score"] = (
        weights.w_promote * upskill_summary["promotion_rate"].fillna(0.0)
        + weights.w_stretch * upskill_summary["mean_delta_score_stretch"].fillna(0.0)
        + weights.w_best * upskill_summary["mean_delta_score_best"].fillna(0.0)
        - weights.w_demote * upskill_summary["demotion_rate"].fillna(0.0)
        - weights.w_tail * tail_penalty
    )

    upskill_summary["passes_guardrail"] = upskill_summary["demotion_rate"].fillna(
        0.0
    ) <= float(demotion_tol)

    ranked = upskill_summary.sort_values(
        by=[
            "passes_guardrail",
            "upskill_impact_score",
            "promotion_rate",
            "mean_delta_score_stretch",
            "mean_delta_score_best",
        ],
        ascending=[False, False, False, False, False],
    )

    upskill_recommendation = ranked.head(int(top_n_skills))
    recommended_scenarios = list(upskill_recommendation.index)

    # -----------------------------
    # 8) Scenario -> family -> example tokens (for report)
    # -----------------------------
    recommendation_dict: dict[str, list[str]] = {}
    for scen in recommended_scenarios:
        fam = scen.replace("upskill_", "", 1)
        recommendation_dict[fam] = _dedup_tokens(missing_dict.get(fam, []), max_n=5)

    # -----------------------------
    # 9) Print report
    # -----------------------------
    if print_report:
        print("Upskilling report")
        print("-----------------")
        print()
        print("Ranking logic:")
        print(
            "- Universe is frozen by job_id (candidate_override_df), so deltas are comparable.\n"
            "- Missing families come from explained stretch jobs (missing_families).\n"
            "- Each scenario injects representative tokens for one family into skill_text.\n"
            "- Scenarios are skipped if the injected tokens do not change the extracted user skill_vector.\n"
            "- Deltas are percentage points (bounded 0–1 metrics × 100).\n"
            "- Ranking rewards promotions + score gains (esp. baseline-stretch), penalises demotions + worst-tail harms.\n"
            f"- Guardrail: demotion_rate <= {demotion_tol}."
        )
        print()
        print(
            f"Top {int(top_n_skills)} recommended skill families (with example tokens):"
        )
        for fam, toks in recommendation_dict.items():
            print(f"* {fam}: {toks}")
        print()
        show_cols = [
            "upskill_impact_score",
            "promotion_rate",
            "demotion_rate",
            "mean_delta_score_stretch",
            "mean_delta_score_best",
            "p10_delta_score_best",
        ]
        print("Top scenarios summary:")
        print(upskill_recommendation[show_cols].round(4))

    return {
        "job_base_upskill": upskill_df,
        "upskill_summary": upskill_summary,
        "upskill_recommendation": upskill_recommendation,
        "missing_dict": missing_dict,
        "recommendation_dict": recommendation_dict,
        "scenario_meta": scenario_meta_df,
        "meta": {
            "tau": float(tau),
            "n_tokens_per_family": int(n_tokens_per_family),
            "top_n_skills": int(top_n_skills),
            "demotion_tol": float(demotion_tol),
            "weights": {
                "w_promote": float(weights.w_promote),
                "w_stretch": float(weights.w_stretch),
                "w_best": float(weights.w_best),
                "w_demote": float(weights.w_demote),
                "w_tail": float(weights.w_tail),
            },
            "constraints": {
                "current_state": current_state,
                "job_title_family": job_title_family,
                "job_title_rich": job_title_rich,
                "target_sectors": target_sectors,
                "salary_target": salary_target,
            },
        },
    }
