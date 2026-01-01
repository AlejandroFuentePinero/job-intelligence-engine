# src/job_intel/app/engine.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple
import os
import subprocess
import sys

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


def _get_git_commit_short() -> str | None:
    """
    Best-effort git commit hash (short). Works in local dev and in GitHub Actions.
    Never raises.
    """
    # CI-first
    sha = (os.environ.get("GITHUB_SHA") or "").strip()
    if sha:
        return sha[:7]

    # Local git repo (best-effort)
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        return out or None
    except Exception:
        return None


def _assets_updated_at_iso(paths: list[Path]) -> str | None:
    """
    Max mtime across a set of artefacts. Returns ISO UTC timestamp or None.
    Never raises.
    """
    try:
        mtimes = [p.stat().st_mtime for p in paths if p.exists()]
        if not mtimes:
            return None
        dt = datetime.fromtimestamp(max(mtimes), tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def get_build_info() -> dict[str, Any]:
    """
    Minimal build-info stamp for UI display (sidebar/footer).
    Safe to call in any environment; never raises.
    """
    repo_root = (
        Path(__file__).resolve().parents[3]
    )  # .../src/job_intel/app/engine.py -> repo root

    # Keep this list small and stable: only what the app/CI smoke expects.
    artefacts = [
        repo_root / "data" / "processed" / "df_with_residuals.csv",
        repo_root
        / "data"
        / "processed"
        / "skill_similarity_matrix"
        / "skill_similarity_edges_k5_embeddings.csv",
        repo_root / "data" / "processed" / "ch5_assets" / "skill_value_index.csv",
        repo_root / "data" / "processed" / "ch5_assets" / "shap_salary_explanation.npz",
        repo_root
        / "data"
        / "processed"
        / "ch5_assets"
        / "fairness_group_summary_long.csv",
        repo_root
        / "data"
        / "processed"
        / "ch5_assets"
        / "fairness_residual_box_stats.json",
    ]

    missing = [str(p.relative_to(repo_root)) for p in artefacts if not p.exists()]
    updated_at = _assets_updated_at_iso(artefacts)

    return {
        "git_commit": _get_git_commit_short(),
        "assets_updated_at_utc": updated_at,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "missing_required_assets": missing,
    }


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
    build_info = get_build_info()

    if not isinstance(profile_or_cfg, dict):
        return AppResult(
            "Invalid input (must be a dict).",
            {"error": "profile_not_dict", "build_info": build_info},
        )

    try:
        user_inputs, pipeline_params, _run_career_sim, scenarios, config = (
            _parse_request(profile_or_cfg)
        )
    except Exception as e:
        return AppResult(
            "Invalid demo/profile payload.",
            {"error": repr(e), "build_info": build_info},
        )

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
        return AppResult(
            "Chapter 4 pipeline error.",
            {"error": repr(e), "build_info": build_info},
        )

    if not isinstance(recommender_out, dict):
        return AppResult(
            "Unexpected recommender_out type.",
            {"type": type(recommender_out).__name__, "build_info": build_info},
        )

    tables = recommender_out.get("tables", {})
    if not isinstance(tables, dict):
        return AppResult(
            "Missing recommender_out['tables'] dict.",
            {"tables_type": type(tables).__name__, "build_info": build_info},
        )

    candidate_jobs = tables.get("candidate_jobs")

    if recommender_expl is None or not isinstance(recommender_expl, dict):
        return AppResult(
            "Explanations missing (expected build_job_explanations output).",
            {"tables_keys": sorted(list(tables.keys())), "build_info": build_info},
        )

    expl_tables = recommender_expl.get("tables", {})
    if not isinstance(expl_tables, dict):
        return AppResult(
            "Explanation tables missing.",
            {"expl_type": type(expl_tables).__name__, "build_info": build_info},
        )

    best_now_expl = expl_tables.get("top_best_explained")
    stretch_expl = expl_tables.get("top_stretch_explained")
    glossary = recommender_expl.get("metric_glossary", {})

    if not isinstance(best_now_expl, pd.DataFrame) or not isinstance(
        stretch_expl, pd.DataFrame
    ):
        return AppResult(
            "Explanation branch ran, but top_best_explained/top_stretch_explained not found.",
            {
                "expl_tables_keys": sorted(list(expl_tables.keys())),
                "build_info": build_info,
            },
        )

    warnings = recommender_out.get("warnings", [])
    warn_n = len(warnings) if isinstance(warnings, list) else 0

    return AppResult(
        f"Recommender ran successfully. Warnings: {warn_n}.",
        {
            "build_info": build_info,
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
