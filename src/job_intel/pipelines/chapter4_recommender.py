# src/job_intel/pipelines/chapter4_recommender.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import logging
import pandas as pd

from src.job_intel.config import MODELS_DIR
from src.job_intel.features.job_recommender import job_recommender
from src.job_intel.features.job_explanations import build_job_explanations
from src.job_intel.features.upskilling_recommender import upskill_recommender

# Canonical simulation contracts + runner (single source of truth)
from src.job_intel.v2_updates.features.career_simulator import (
    SimulationScenario,
    SimulationConfig,
    career_simulation,
)

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Avoid double-handlers when called repeatedly (e.g., notebooks)."""
    if verbose and not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
        )
    if not verbose:
        logger.setLevel(logging.WARNING)


# -----------------------------
# Local contracts (pipeline-only)
# -----------------------------
@dataclass(frozen=True)
class UpskillRankingWeights:
    """Weights for the composite ranking score in upskilling (pass-through)."""

    w_promote: float = 5.0
    w_stretch: float = 1.0
    w_best: float = 0.5
    w_demote: float = 10.0
    w_tail: float = 1.0


@dataclass(frozen=True)
class Chapter4PipelineOutput:
    """Optional stable container (pipeline still returns a legacy tuple)."""

    recommender_out: dict
    recommender_explanation: Optional[dict]
    upskilling_out: Optional[dict]
    career_sim: Optional[dict]


# -----------------------------
# Validation helpers
# -----------------------------
def _as_list_or_none(x: Optional[Sequence[str]]) -> Optional[list[str]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    raise TypeError(f"Expected list/tuple[str] or None, got {type(x)}")


def _validate_weights(w_skill: float, w_salary: float) -> tuple[float, float]:
    for name, w in (("w_skill", w_skill), ("w_salary", w_salary)):
        if not isinstance(w, (int, float)):
            raise TypeError(f"{name} must be numeric, got {type(w)}")
        if w < 0:
            raise ValueError(f"{name} must be >= 0, got {w}")
    s = float(w_skill + w_salary)
    if s == 0:
        raise ValueError("w_skill + w_salary must be > 0")
    return float(w_skill / s), float(w_salary / s)


def _validate_nonneg_int(name: str, v: Optional[int], allow_none: bool = False) -> None:
    if v is None and allow_none:
        return
    if not isinstance(v, int):
        raise TypeError(f"{name} must be int, got {type(v)}")
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")


def _validate_fraction(name: str, v: float) -> None:
    if not isinstance(v, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(v)}")
    if not (0.0 <= float(v) <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {v}")


def _validate_pathlike(p: object, name: str) -> Path:
    if isinstance(p, Path):
        return p
    if isinstance(p, str):
        return Path(p)
    raise TypeError(f"{name} must be str or Path, got {type(p)}")


def _validate_scenarios(
    scenarios: Optional[list[SimulationScenario]],
) -> Optional[list[SimulationScenario]]:
    """
    Validate scenarios using the *canonical* SimulationScenario imported from career_simulator.
    Also normalizes tokens (trim/lowers/dedup) within max_tokens.
    """
    if scenarios is None:
        return None
    if not isinstance(scenarios, list):
        raise TypeError(
            f"scenarios must be list[SimulationScenario] or None, got {type(scenarios)}"
        )

    def _clean_tokens(tokens: Sequence[str], max_tokens: int) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for t in tokens:
            s = str(t).strip().lower()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= int(max_tokens):
                break
        return out

    validated: list[SimulationScenario] = []
    for sc in scenarios:
        if not isinstance(sc, SimulationScenario):
            raise TypeError(f"Each scenario must be SimulationScenario, got {type(sc)}")

        toks = _clean_tokens(sc.added_tokens, sc.max_tokens)
        validated.append(
            SimulationScenario(
                name=sc.name,
                added_tokens=toks,
                max_tokens=sc.max_tokens,
                demotion_tol=sc.demotion_tol,
            )
        )

    # dedupe by scenario name (keep first)
    seen_names: set[str] = set()
    out: list[SimulationScenario] = []
    for sc in validated:
        if sc.name in seen_names:
            logger.warning(
                "Duplicate scenario name '%s' found; keeping first occurrence.", sc.name
            )
            continue
        seen_names.add(sc.name)
        out.append(sc)
    return out


# -----------------------------
# Pipeline
# -----------------------------
def run_recommender_pipeline(
    # User input control
    skill_text: str = "",
    current_state: str = "ALL",
    job_title_family: str | None = None,
    job_title_rich: str | None = None,
    target_sectors: list[str] | None = None,
    salary_target: float | None = None,
    explain_skills: bool | None = None,
    w_skill: float = 0.7,
    w_salary: float = 0.3,
    top_k_gaps: int = 200,
    return_top_n_jobs: int | None = 6200,  # None for full-universe engine mode
    run_sensitivity: bool = False,
    salary_model_path: str | Path = MODELS_DIR / "salary_model_v4.pkl",
    # Shared override
    candidate_override_df: Optional[pd.DataFrame] = None,
    # Recommender control
    s_min_base: float = 0.70,
    s_min_floor: float = 0.60,
    n_target: int = 50,
    c_max: float = 0.50,
    min_bucket_size_bestnow: int = 10,
    min_bucket_size_stretch: int = 5,
    alpha: float = 0.5,
    top_n_best: int = 10,
    top_n_stretch: int = 5,
    verbose: bool = True,
    # Explanation controller
    run_explanator: bool = True,
    tau: float = 0.50,
    validate: bool = True,
    include_scored_universe: bool = True,
    # Upskilling controller
    run_upskilling: bool = True,
    n_tokens_per_family: int = 3,
    weights: UpskillRankingWeights = UpskillRankingWeights(),
    demotion_tol: float = 0.0,
    top_n_skills: int = 3,
    print_report: bool = True,
    # Simulation-specific
    run_career_sim: bool = False,
    scenarios: list[SimulationScenario] | None = None,
    config: SimulationConfig = SimulationConfig(),
):
    """
    Chapter 4 orchestrator pipeline.

    Returns legacy tuple for backward compatibility:
      (recommender_out, recommender_explanation, upskilling_out, career_sim)
    """
    _configure_logging(verbose)

    # ---- validation / normalization ----
    if skill_text is None:
        skill_text = ""
    if not isinstance(skill_text, str):
        raise TypeError(f"skill_text must be str, got {type(skill_text)}")
    if not isinstance(current_state, str) or not current_state:
        raise ValueError("current_state must be a non-empty string (or 'ALL').")

    target_sectors = _as_list_or_none(target_sectors)

    if salary_target is not None:
        if not isinstance(salary_target, (int, float)):
            raise TypeError(
                f"salary_target must be numeric or None, got {type(salary_target)}"
            )
        if float(salary_target) <= 0:
            raise ValueError("salary_target must be > 0 when provided.")

    w_skill, w_salary = _validate_weights(w_skill, w_salary)

    _validate_nonneg_int("top_k_gaps", top_k_gaps)
    _validate_nonneg_int("return_top_n_jobs", return_top_n_jobs, allow_none=True)
    _validate_nonneg_int("n_target", n_target)
    _validate_nonneg_int("min_bucket_size_bestnow", min_bucket_size_bestnow)
    _validate_nonneg_int("min_bucket_size_stretch", min_bucket_size_stretch)
    _validate_nonneg_int("top_n_best", top_n_best)
    _validate_nonneg_int("top_n_stretch", top_n_stretch)

    _validate_fraction("s_min_base", s_min_base)
    _validate_fraction("s_min_floor", s_min_floor)
    _validate_fraction("c_max", c_max)
    _validate_fraction("tau", tau)
    _validate_fraction("demotion_tol", demotion_tol)

    if not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be numeric, got {type(alpha)}")
    if float(alpha) < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")

    salary_model_path = _validate_pathlike(salary_model_path, "salary_model_path")

    if candidate_override_df is not None:
        if not isinstance(candidate_override_df, pd.DataFrame):
            raise TypeError("candidate_override_df must be a pandas DataFrame or None")
        if "job_id" not in candidate_override_df.columns:
            raise ValueError("candidate_override_df must contain a 'job_id' column")
        if candidate_override_df["job_id"].isna().any():
            raise ValueError("candidate_override_df['job_id'] contains NaNs")

    scenarios = _validate_scenarios(scenarios)
    if run_career_sim and not scenarios:
        raise ValueError("run_career_sim=True requires a non-empty 'scenarios' list.")

    # ---- baseline recommender ----
    logger.info("Running Chapter 4 recommender...")
    recommender_out = job_recommender(
        skill_text=skill_text,
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
        salary_model_path=salary_model_path,
        candidate_override_df=candidate_override_df,
        s_min_base=s_min_base,
        s_min_floor=s_min_floor,
        n_target=n_target,
        c_max=c_max,
        min_bucket_size_bestnow=min_bucket_size_bestnow,
        min_bucket_size_stretch=min_bucket_size_stretch,
        alpha=alpha,
        top_n_best=top_n_best,
        top_n_stretch=top_n_stretch,
        verbose=verbose,
    )

    # ---- explanations ----
    recommender_explanation: Optional[dict] = None
    if run_explanator:
        logger.info("Building explanations (tau=%.2f)...", tau)
        recommender_explanation = build_job_explanations(
            rec=recommender_out,
            tau=tau,
            validate=validate,
            include_scored_universe=include_scored_universe,
        )

    # ---- upskilling ----
    upskilling_out: Optional[dict] = None
    if run_upskilling:
        _validate_nonneg_int("n_tokens_per_family", n_tokens_per_family)
        _validate_nonneg_int("top_n_skills", top_n_skills)

        logger.info("Running upskilling recommender (top_n_skills=%d)...", top_n_skills)
        # NOTE: upskill_recommender currently freezes universe internally.
        # Do NOT pass candidate_override_df unless the function signature supports it.
        upskilling_out = upskill_recommender(
            skill_text=skill_text,
            current_state=current_state,
            job_title_family=job_title_family,
            job_title_rich=job_title_rich,
            target_sectors=target_sectors,
            salary_target=salary_target,
            explain_skills=explain_skills,
            tau=tau,
            n_tokens_per_family=n_tokens_per_family,
            weights=weights,
            demotion_tol=demotion_tol,
            top_n_skills=top_n_skills,
            w_skill=w_skill,
            w_salary=w_salary,
            top_k_gaps=top_k_gaps,
            return_top_n_jobs=return_top_n_jobs,
            run_sensitivity=run_sensitivity,
            print_report=print_report,
        )

    # ---- career simulation ----
    career_sim: Optional[dict] = None
    if run_career_sim:
        logger.info(
            "Running career simulation (n_scenarios=%d)...", len(scenarios or [])
        )
        career_sim = career_simulation(
            skill_text=skill_text,
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
            scenarios=scenarios,
            config=config,
        )

    out = Chapter4PipelineOutput(
        recommender_out=recommender_out,
        recommender_explanation=recommender_explanation,
        upskilling_out=upskilling_out,
        career_sim=career_sim,
    )

    return (
        out.recommender_out,
        out.recommender_explanation,
        out.upskilling_out,
        out.career_sim,
    )
