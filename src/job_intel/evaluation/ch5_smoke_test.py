# src/job_intel/evaluation/ch5_smoke_test.py
"""
Chapter 5 — Smoke Test (v1)

Purpose:
- Prove end-to-end wiring is intact:
  1) (optional) build + validate Chapter 5 assets
  2) load demo persona config
  3) run Chapter 4 recommender pipeline once (explanations + upskilling enabled; career sim disabled)
  4) assert key tables exist and are non-empty

Run (from repo root):
  python -m src.job_intel.evaluation.ch5_smoke_test

Optional:
  python -m src.job_intel.evaluation.ch5_smoke_test --skip-build
  python -m src.job_intel.evaluation.ch5_smoke_test --check-determinism
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.job_intel.pipelines.chapter4_recommender import run_recommender_pipeline

# Always disable career simulation for app-like runs
from src.job_intel.v2_updates.features.career_simulator import SimulationConfig


@dataclass(frozen=True)
class SmokeResult:
    best_now_rows: int
    stretch_rows: int
    candidate_rows: int
    best_now_job_ids: list[str]
    stretch_job_ids: list[str]


def _repo_root() -> Path:
    # .../src/job_intel/evaluation/ch5_smoke_test.py -> repo root = parents[3]
    return Path(__file__).resolve().parents[3]


def _demo_path() -> Path:
    return _repo_root() / "src" / "job_intel" / "evaluation" / "recommender_demo.json"


def _load_demo_cfg(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Demo config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Demo config must be a dict; got {type(cfg)}")
    return cfg


def _split_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "user_inputs" in payload:
        user_inputs = payload.get("user_inputs") or {}
        pipeline_params = payload.get("pipeline_params") or {}
        if not isinstance(user_inputs, dict) or not isinstance(pipeline_params, dict):
            raise TypeError("demo_cfg user_inputs/pipeline_params must be dicts")
        return user_inputs, pipeline_params
    return payload, {}


def _run_pipeline_from_demo(cfg: dict[str, Any]) -> SmokeResult:
    user_inputs, pipeline_params = _split_payload(cfg)

    # Copy then force app-like defaults (fast + consistent)
    pipeline_params = dict(pipeline_params or {})
    pipeline_params.pop("run_career_sim", None)
    pipeline_params.pop("scenarios", None)
    pipeline_params.pop("config", None)

    pipeline_params["run_explanator"] = True
    pipeline_params["run_upskilling"] = True

    rec_out, expl_out, up_out, sim_out = run_recommender_pipeline(
        **user_inputs,
        **pipeline_params,
        run_career_sim=False,
        scenarios=None,
        config=SimulationConfig(),
    )

    # --- extract tables in the same priority order as the app ---
    best_df = None
    stretch_df = None

    if isinstance(expl_out, dict):
        tables = expl_out.get("tables", {}) or {}
        best_df = tables.get("top_best_explained")
        stretch_df = tables.get("top_stretch_explained")

    if best_df is None and isinstance(rec_out, dict):
        best_df = (rec_out.get("tables", {}) or {}).get("top_best_now")

    if stretch_df is None and isinstance(rec_out, dict):
        stretch_df = (rec_out.get("tables", {}) or {}).get("top_stretch")

    candidate_df = None
    if isinstance(rec_out, dict):
        candidate_df = (rec_out.get("tables", {}) or {}).get("candidate_jobs")

    # --- assertions (fail-fast) ---
    if not isinstance(best_df, pd.DataFrame) or best_df.empty:
        raise AssertionError("Smoke test failed: best_now table missing or empty.")

    if not isinstance(stretch_df, pd.DataFrame) or stretch_df.empty:
        raise AssertionError("Smoke test failed: stretch table missing or empty.")

    if not isinstance(candidate_df, pd.DataFrame) or candidate_df.empty:
        raise AssertionError(
            "Smoke test failed: candidate_jobs table missing or empty."
        )

    if "job_id" not in best_df.columns and best_df.index.name != "job_id":
        raise AssertionError("Smoke test failed: best_now table missing job_id.")
    if "job_id" not in stretch_df.columns and stretch_df.index.name != "job_id":
        raise AssertionError("Smoke test failed: stretch table missing job_id.")

    # Normalize job_ids for determinism check / reporting
    def _job_ids(df: pd.DataFrame) -> list[str]:
        if "job_id" in df.columns:
            s = df["job_id"]
        elif df.index.name == "job_id":
            s = df.index.to_series()
        else:
            s = pd.Series([], dtype=str)
        return s.dropna().astype(str).tolist()

    best_ids = _job_ids(best_df)
    stretch_ids = _job_ids(stretch_df)

    if len(best_ids) == 0 or len(stretch_ids) == 0:
        raise AssertionError(
            "Smoke test failed: could not extract job_ids from outputs."
        )

    # Optional light sanity check on module outputs
    if sim_out is not None:
        raise AssertionError(
            "Smoke test failed: career_sim output should be None in v1 app runs."
        )
    if up_out is None:
        raise AssertionError(
            "Smoke test failed: upskilling output missing (expected enabled)."
        )

    return SmokeResult(
        best_now_rows=int(best_df.shape[0]),
        stretch_rows=int(stretch_df.shape[0]),
        candidate_rows=int(candidate_df.shape[0]),
        best_now_job_ids=best_ids,
        stretch_job_ids=stretch_ids,
    )


def _maybe_build_assets(skip_build: bool) -> None:
    if skip_build:
        return
    # Build + validate ch5 assets (fail-fast if missing)
    from src.job_intel.pipelines.ch5_app_build import build_ch5_assets

    build_ch5_assets(rebuild_fairness=True, strict=True, validate_demo_cfg=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chapter 5 smoke test (v1)")
    p.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip ch5 asset build/validation step (only run pipeline smoke test).",
    )
    p.add_argument(
        "--check-determinism",
        action="store_true",
        help="Run the demo pipeline twice and assert the top job_ids match.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    argv = argv if argv is not None else sys.argv[1:]
    args = _parse_args(argv)

    _maybe_build_assets(skip_build=bool(args.skip_build))

    cfg = _load_demo_cfg(_demo_path())

    r1 = _run_pipeline_from_demo(cfg)
    print(
        "\nSmoke test (run 1) OK:"
        f"\n  best_now rows: {r1.best_now_rows}"
        f"\n  stretch rows:  {r1.stretch_rows}"
        f"\n  candidates:    {r1.candidate_rows}"
    )

    if args.check_determinism:
        r2 = _run_pipeline_from_demo(cfg)
        if r1.best_now_job_ids[:20] != r2.best_now_job_ids[:20]:
            raise AssertionError(
                "Determinism failed: best_now job_ids differ across runs."
            )
        if r1.stretch_job_ids[:20] != r2.stretch_job_ids[:20]:
            raise AssertionError(
                "Determinism failed: stretch job_ids differ across runs."
            )
        print("\nDeterminism check OK (top 20 job_ids match across runs).")

    return {
        "status": "ok",
        "best_now_rows": r1.best_now_rows,
        "stretch_rows": r1.stretch_rows,
        "candidate_rows": r1.candidate_rows,
    }


if __name__ == "__main__":
    main()
