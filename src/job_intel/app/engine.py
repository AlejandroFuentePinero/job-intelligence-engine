# src/job_intel/app/engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import pandas as pd

from src.job_intel.pipelines.chapter4_recommender import run_recommender_pipeline
from src.job_intel.v2_updates.features.career_simulator import (
    SimulationScenario,
    SimulationConfig,
)


@dataclass
class AppResult:
    narrative: str
    payload: dict[str, Any]


def _parse_request(
    profile_or_cfg: dict[str, Any],
) -> Tuple[
    dict[str, Any], dict[str, Any], bool, list[SimulationScenario], SimulationConfig
]:
    """
    Accepts either:
      - a demo config dict with keys: user_inputs, pipeline_params, career_simulation
      - a manual profile dict with direct user fields
    """
    if "user_inputs" in profile_or_cfg:
        cfg = profile_or_cfg
        user_inputs = cfg.get("user_inputs", {}) or {}
        pipeline_params = cfg.get("pipeline_params", {}) or {}

        sim_block = cfg.get("career_simulation", {}) or {}
        run_career_sim = bool(sim_block.get("run_career_sim", False))
        scenarios_raw = sim_block.get("scenarios", []) or []
        scenarios = [SimulationScenario(**d) for d in scenarios_raw]
        config = (
            SimulationConfig(**sim_block.get("config", {}))
            if sim_block.get("config")
            else SimulationConfig()
        )
        return user_inputs, pipeline_params, run_career_sim, scenarios, config

    # Manual mode
    user_inputs = {
        "skill_text": profile_or_cfg.get("skill_text", "") or "",
        "current_state": profile_or_cfg.get("current_state", "ALL") or "ALL",
        "job_title_family": profile_or_cfg.get("job_title_family", None),
        "job_title_rich": profile_or_cfg.get("job_title_rich", None),
        "target_sectors": profile_or_cfg.get("target_sectors", None),
        "salary_target": profile_or_cfg.get("salary_target", None),
    }
    return user_inputs, {}, False, [], SimulationConfig()


def run_recommender(profile_or_cfg: dict[str, Any]) -> AppResult:
    """
    App runner:
      - forces run_career_sim=False (fast)
      - forces run_explanator=True (use explanation branch)
      - returns explained tables + glossary + candidate_jobs
    """
    if not isinstance(profile_or_cfg, dict):
        return AppResult(
            "Invalid input (must be a dict).", {"error": "profile_not_dict"}
        )

    try:
        user_inputs, pipeline_params, _run_career_sim, scenarios, config = (
            _parse_request(profile_or_cfg)
        )
    except Exception as e:
        return AppResult("Invalid demo/profile payload.", {"error": repr(e)})

    # --- hard app constraints (fast + deterministic) ---
    pp = dict(pipeline_params)

    # Remove any keys that would conflict with explicit arguments or slow the app
    pp.pop("run_career_sim", None)
    pp.pop("scenarios", None)
    pp.pop("config", None)

    # Force these for the app
    pp["run_explanator"] = True
    pp["run_upskilling"] = True
    pp["print_report"] = False
    pp["verbose"] = False
    # If your pipeline supports it, keep it light:
    pp["include_scored_universe"] = False

    try:
        recommender_out, recommender_expl, upskilling_out, career_sim = (
            run_recommender_pipeline(
                **user_inputs,
                **pp,
                run_career_sim=False,
                scenarios=None,
                config=SimulationConfig(),
            )
        )
    except Exception as e:
        return AppResult("Chapter 4 pipeline error.", {"error": repr(e)})

    if not isinstance(recommender_out, dict):
        return AppResult(
            "Unexpected recommender_out type.", {"type": type(recommender_out).__name__}
        )

    tables = recommender_out.get("tables", {})
    if not isinstance(tables, dict):
        return AppResult(
            "Missing recommender_out['tables'] dict.",
            {"tables_type": type(tables).__name__},
        )

    candidate_jobs = tables.get("candidate_jobs")

    if recommender_expl is None or not isinstance(recommender_expl, dict):
        return AppResult(
            "Explanations missing (expected build_job_explanations output).",
            {"tables_keys": sorted(list(tables.keys()))},
        )

    expl_tables = recommender_expl.get("tables", {})
    if not isinstance(expl_tables, dict):
        return AppResult(
            "Explanation tables missing.", {"expl_type": type(expl_tables).__name__}
        )

    best_now_expl = expl_tables.get("top_best_explained")
    stretch_expl = expl_tables.get("top_stretch_explained")
    glossary = recommender_expl.get("metric_glossary", {})

    if not isinstance(best_now_expl, pd.DataFrame) or not isinstance(
        stretch_expl, pd.DataFrame
    ):
        return AppResult(
            "Explanation branch ran, but top_best_explained/top_stretch_explained not found.",
            {"expl_tables_keys": sorted(list(expl_tables.keys()))},
        )

    warnings = recommender_out.get("warnings", [])
    warn_n = len(warnings) if isinstance(warnings, list) else 0

    return AppResult(
        f"Recommender ran successfully. Warnings: {warn_n}.",
        {
            "best_now_explained": best_now_expl,
            "stretch_explained": stretch_expl,
            "candidate_jobs": (
                candidate_jobs if isinstance(candidate_jobs, pd.DataFrame) else None
            ),
            "metric_glossary": glossary if isinstance(glossary, dict) else {},
            "warnings": warnings,
            "counts": recommender_out.get("counts", {}),
            "salary_summary": recommender_out.get("salary_summary", {}),
        },
    )
