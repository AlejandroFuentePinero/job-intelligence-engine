# src/job_intel/features/career_simulator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.job_intel.features.job_recommender import job_recommender


# -----------------------------
# Data contracts
# -----------------------------
ScenarioStatus = Literal["kept", "skipped"]


@dataclass(frozen=True)
class SimulationScenario:
    """
    A single what-if scenario: add tokens to the user's skill_text, re-run on frozen universe,
    and compute deltas vs baseline.
    """

    name: str
    added_tokens: list[str]  # e.g. ["sql", "postgres"]
    max_tokens: int = 3  # guardrail: keep scenarios small
    demotion_tol: float = 0.0  # scenario-specific guardrail


@dataclass(frozen=True)
class SimulationConfig:
    """
    Controls output size and safety checks.
    """

    top_n_unlocked_jobs: int = 20
    require_frozen_universe: bool = True
    enforce_same_job_set: bool = True  # strict: same job_id set per scenario


# -----------------------------
# Core helpers (mirrors upskilling_recommender patterns)
# -----------------------------
METRIC_COLS = [
    "skill_match_norm",
    "suitability",
    "expected_missing_norm",
    "competitiveness_index",
    "bucket",
    "score",
]


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


def _require_cols(df: pd.DataFrame, cols: list[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where} is missing required columns: {missing}")


def _get_skill_vector(rec: dict[str, Any]) -> pd.DataFrame | None:
    """
    Return profile['derived']['skill_vector'] as a single-row DataFrame if present, else None.
    Used for no-op detection when scenarios add unrecognized tokens.
    """
    try:
        prof = rec["profile"]
        vec = prof["derived"]["skill_vector"]
        if isinstance(vec, pd.DataFrame) and len(vec) == 1:
            return vec
        return None
    except Exception:
        return None


def _baseline_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map job_id -> baseline_* metrics for delta merges.
    """
    cols = ["job_id"] + METRIC_COLS
    _require_cols(df, cols, where="_baseline_map input df")

    base = df[cols].copy()
    rename = {c: f"baseline_{c}" for c in METRIC_COLS}
    base = base.rename(columns=rename)
    return base


def _bucket_movement(baseline_bucket: pd.Series, new_bucket: pd.Series) -> pd.Series:
    """
    promoted: stretch -> best_now
    demoted: best_now -> stretch
    unchanged: else
    """
    base = baseline_bucket.astype(str)
    new = new_bucket.astype(str)

    promoted = (base == "stretch") & (new == "best_now")
    demoted = (base == "best_now") & (new == "stretch")

    return np.where(
        promoted,
        "promoted",
        np.where(demoted, "demoted", "unchanged"),
    )


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


# -----------------------------
# Public API
# -----------------------------
def career_simulation(
    *,
    # same user-facing inputs as job_recommender
    skill_text: str = "",
    current_state: str = "ALL",
    job_title_family: str | None = None,
    job_title_rich: str | None = None,
    target_sectors: list[str] | None = None,
    salary_target: float | None = None,
    explain_skills: bool | None = None,
    # recommender controls
    w_skill: float = 0.7,
    w_salary: float = 0.3,
    top_k_gaps: int = 200,
    return_top_n_jobs: int | None = 6200,
    run_sensitivity: bool = False,
    # simulation-specific
    scenarios: list[SimulationScenario] | None = None,
    config: SimulationConfig = SimulationConfig(),
) -> dict[str, Any]:
    """
    Minimal career simulation:
      1) baseline run
      2) freeze universe by job_id
      3) run scenarios on frozen universe
      4) compute per-job deltas + scenario summaries + unlocked jobs
    """
    if not scenarios:
        raise ValueError("Provide at least one SimulationScenario.")
    if int(config.top_n_unlocked_jobs) <= 0:
        raise ValueError("config.top_n_unlocked_jobs must be >= 1")
    if not (0.0 <= float(w_skill) <= 1.0) or not (0.0 <= float(w_salary) <= 1.0):
        raise ValueError("w_skill and w_salary must be in [0,1]")
    if abs(float(w_skill) + float(w_salary) - 1.0) > 1e-6:
        raise ValueError("w_skill + w_salary must sum to 1.0")

    base_skill_text = (skill_text or "").strip()
    base_skill_text_lc = base_skill_text.lower()

    # ---- 1) Baseline run (unfrozen)
    baseline_rec = job_recommender(
        skill_text=base_skill_text,
        current_state=current_state,
        job_title_family=job_title_family,
        job_title_rich=job_title_rich,
        target_sectors=target_sectors,
        salary_target=salary_target,
        explain_skills=explain_skills,
        w_skill=w_skill,
        w_salary=w_salary,
        top_k_gaps=top_k_gaps,
        return_top_n_jobs=return_top_n_jobs,
        run_sensitivity=run_sensitivity,
        candidate_override_df=None,
        verbose=False,
    )

    base_universe = baseline_rec["tables"]["scored_universe"].copy()
    _require_cols(
        base_universe,
        ["job_id"] + METRIC_COLS,
        where="baseline scored_universe",
    )

    # freeze universe by job_id
    override_df = base_universe[["job_id"]].copy()
    baseline_job_ids = set(override_df["job_id"].tolist())
    baseline_n = int(len(override_df))

    # identity columns for nicer reports (optional)
    identity_candidates = [
        "job_title",
        "job_title_clean",
        "Job Title",
        "company_name",
        "company",
        "Company",
        "state",
        "location",
        "sector",
        "industry",
    ]
    identity_cols = [c for c in identity_candidates if c in base_universe.columns]
    identity_df = base_universe[["job_id"] + identity_cols].drop_duplicates("job_id")

    # baseline skill-vector (for no-op detection)
    baseline_vec = _get_skill_vector(baseline_rec)
    if baseline_vec is not None:
        baseline_vec = baseline_vec.reindex(sorted(baseline_vec.columns), axis=1)

    base_map = _baseline_map(base_universe)

    # Optional: re-run baseline under frozen universe so baseline/scenarios are 1:1 consistent
    if config.require_frozen_universe:
        baseline_frozen_rec = job_recommender(
            skill_text=base_skill_text,
            current_state=current_state,
            job_title_family=job_title_family,
            job_title_rich=job_title_rich,
            target_sectors=target_sectors,
            salary_target=salary_target,
            explain_skills=explain_skills,
            w_skill=w_skill,
            w_salary=w_salary,
            top_k_gaps=top_k_gaps,
            return_top_n_jobs=None,  # IMPORTANT when override is provided
            run_sensitivity=run_sensitivity,
            candidate_override_df=override_df,
            verbose=False,
        )
        base_universe = baseline_frozen_rec["tables"]["scored_universe"].copy()
        _require_cols(
            base_universe,
            ["job_id"] + METRIC_COLS,
            where="baseline frozen scored_universe",
        )
        base_map = _baseline_map(base_universe)

        # update identity_df to match frozen baseline if possible
        if identity_cols:
            identity_df = base_universe[["job_id"] + identity_cols].drop_duplicates(
                "job_id"
            )

    # ---- 2) Run scenarios (frozen universe)
    scenario_tables: list[pd.DataFrame] = []
    scenario_meta: list[dict[str, Any]] = []

    for s in scenarios:
        name = str(s.name).strip()
        if not name:
            raise ValueError("SimulationScenario.name cannot be empty")

        toks = _dedup_tokens(s.added_tokens, max_n=s.max_tokens)
        toks = [t for t in toks if t.lower() not in base_skill_text_lc]

        if not toks:
            scenario_meta.append(
                {
                    "scenario": name,
                    "status": "skipped",
                    "reason": "no_tokens",
                    "tokens": [],
                }
            )
            continue

        scen_skill_text = (base_skill_text + " " + " ".join(toks)).strip()

        scen_rec = job_recommender(
            skill_text=scen_skill_text,
            current_state=current_state,
            job_title_family=job_title_family,
            job_title_rich=job_title_rich,
            target_sectors=target_sectors,
            salary_target=salary_target,
            explain_skills=explain_skills,
            w_skill=w_skill,
            w_salary=w_salary,
            top_k_gaps=top_k_gaps,
            return_top_n_jobs=None,  # frozen
            run_sensitivity=run_sensitivity,
            candidate_override_df=override_df,
            verbose=False,
        )

        # no-op detection: if the scenario doesn't change extracted skill_vector, skip it
        if baseline_vec is not None:
            scen_vec = _get_skill_vector(scen_rec)
            if scen_vec is None:
                scenario_meta.append(
                    {
                        "scenario": name,
                        "status": "skipped",
                        "reason": "missing_skill_vector",
                        "tokens": toks,
                    }
                )
                continue
            scen_vec = scen_vec.reindex(sorted(scen_vec.columns), axis=1)
            if scen_vec.equals(baseline_vec):
                scenario_meta.append(
                    {
                        "scenario": name,
                        "status": "skipped",
                        "reason": "skill_vector_no_effect",
                        "tokens": toks,
                    }
                )
                continue

        scen_uni = scen_rec["tables"]["scored_universe"].copy()
        _require_cols(
            scen_uni,
            ["job_id"] + METRIC_COLS,
            where=f"scenario {name} scored_universe",
        )

        # strict universe checks per scenario
        if int(len(scen_uni)) != baseline_n:
            scenario_meta.append(
                {
                    "scenario": name,
                    "status": "skipped",
                    "reason": f"universe_size_changed (got {len(scen_uni)}, expected {baseline_n})",
                    "tokens": toks,
                }
            )
            continue

        if config.enforce_same_job_set:
            scen_job_ids = set(scen_uni["job_id"].tolist())
            if scen_job_ids != baseline_job_ids:
                scenario_meta.append(
                    {
                        "scenario": name,
                        "status": "skipped",
                        "reason": "job_id_set_changed",
                        "tokens": toks,
                    }
                )
                continue

        scen_uni = scen_uni[["job_id"] + METRIC_COLS].copy()
        scen_uni["scenario"] = name
        scenario_tables.append(scen_uni)

        scenario_meta.append(
            {
                "scenario": name,
                "status": "kept",
                "reason": None,
                "tokens": toks,
                "demotion_tol": float(s.demotion_tol),
            }
        )

    if not scenario_tables:
        raise ValueError("All scenarios were skipped; nothing to simulate.")

    scenario_jobs = pd.concat(scenario_tables, ignore_index=True)

    # ---- 3) Compute deltas
    deltas = scenario_jobs.merge(
        base_map, on="job_id", how="left", validate="many_to_one"
    )

    # merge failure check
    req_baseline_cols = [f"baseline_{c}" for c in METRIC_COLS]
    if deltas[req_baseline_cols].isna().any().any():
        bad = deltas.loc[deltas["baseline_bucket"].isna(), "job_id"].head(10).tolist()
        raise ValueError(
            "Baseline merge failed for some job_id rows; check job_id alignment. "
            f"Example job_ids: {bad}"
        )

    # deltas are expressed as percentage-points (bounded metrics × 100) to match your Chapter 4 style
    deltas["delta_skill_match_norm"] = (
        deltas["skill_match_norm"] - deltas["baseline_skill_match_norm"]
    ) * 100.0
    deltas["delta_suitability"] = (
        deltas["suitability"] - deltas["baseline_suitability"]
    ) * 100.0
    deltas["delta_expected_missing_norm"] = (
        deltas["expected_missing_norm"] - deltas["baseline_expected_missing_norm"]
    ) * 100.0
    deltas["delta_competitiveness_index"] = (
        deltas["competitiveness_index"] - deltas["baseline_competitiveness_index"]
    ) * 100.0
    deltas["delta_score"] = (deltas["score"] - deltas["baseline_score"]) * 100.0

    deltas["bucket_movement"] = _bucket_movement(
        baseline_bucket=deltas["baseline_bucket"],
        new_bucket=deltas["bucket"],
    )

    # attach identity columns if available
    if identity_cols:
        deltas = deltas.merge(
            identity_df, on="job_id", how="left", validate="many_to_one"
        )

    # ---- 4) Scenario summaries
    meta_df = pd.DataFrame(scenario_meta)
    kept_meta = meta_df.loc[meta_df["status"] == "kept"].copy()
    kept_meta = kept_meta[["scenario", "demotion_tol"]].drop_duplicates("scenario")

    rows: list[dict[str, Any]] = []
    for scen, g in deltas.groupby("scenario", sort=False):
        g_best0 = g[g["baseline_bucket"] == "best_now"]
        g_stretch0 = g[g["baseline_bucket"] == "stretch"]

        n_best0 = int(len(g_best0))
        n_stretch0 = int(len(g_stretch0))

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
                "scenario": scen,
                "n_best0": n_best0,
                "n_stretch0": n_stretch0,
                "n_promoted": n_promoted,
                "promotion_rate": promotion_rate,
                "n_demoted": n_demoted,
                "demotion_rate": demotion_rate,
                "mean_delta_score_best": _mean(g_best0["delta_score"]),
                "mean_delta_score_stretch": _mean(g_stretch0["delta_score"]),
                "p10_delta_score_best": _quantile_10(g_best0["delta_score"]),
                "mean_delta_comp_best": _mean(g_best0["delta_competitiveness_index"]),
                "mean_delta_comp_stretch": _mean(
                    g_stretch0["delta_competitiveness_index"]
                ),
            }
        )

    scenario_summary = pd.DataFrame(rows).merge(kept_meta, on="scenario", how="left")

    # guardrail flag (scenario-specific tol)
    scenario_summary["demotion_tol"] = scenario_summary["demotion_tol"].fillna(0.0)
    scenario_summary["passes_guardrail"] = scenario_summary["demotion_rate"].fillna(
        0.0
    ) <= scenario_summary["demotion_tol"].astype(float)

    # tail risk proxy (negative worst-tail on baseline-best jobs)
    tail_penalty = (
        scenario_summary["p10_delta_score_best"]
        .where(scenario_summary["p10_delta_score_best"] < 0, 0.0)
        .abs()
        .fillna(0.0)
    )
    scenario_summary["tail_penalty"] = tail_penalty

    # sort for human usability (guardrail pass first)
    scenario_summary = scenario_summary.sort_values(
        by=[
            "passes_guardrail",
            "promotion_rate",
            "mean_delta_score_stretch",
            "mean_delta_score_best",
            "tail_penalty",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    # ---- 5) Top unlocked jobs (promoted)
    promoted = deltas.loc[deltas["bucket_movement"] == "promoted"].copy()
    promoted = promoted.sort_values(
        ["scenario", "delta_score"], ascending=[True, False]
    )

    # take top N per scenario
    top_unlocked_jobs = (
        promoted.groupby("scenario", as_index=False, sort=False)
        .head(int(config.top_n_unlocked_jobs))
        .reset_index(drop=True)
    )

    # return a compact view by default (keep full deltas separately)
    return {
        "baseline_universe": base_universe,
        "scenario_jobs": scenario_jobs,
        "deltas": deltas,
        "scenario_summary": scenario_summary,
        "top_unlocked_jobs": top_unlocked_jobs,
        "scenario_meta": meta_df,
        "meta": {
            "constraints": {
                "current_state": current_state,
                "job_title_family": job_title_family,
                "job_title_rich": job_title_rich,
                "target_sectors": target_sectors,
                "salary_target": salary_target,
            },
            "config": {
                "top_n_unlocked_jobs": int(config.top_n_unlocked_jobs),
                "require_frozen_universe": bool(config.require_frozen_universe),
                "enforce_same_job_set": bool(config.enforce_same_job_set),
            },
        },
    }
