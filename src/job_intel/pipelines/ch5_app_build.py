# src/job_intel/pipelines/ch5_app_build.py
"""
Chapter 5 — App Build (deterministic, no training)

Goal:
- Build the minimal persisted assets that the Chapter 5 Streamlit app loads fast.
- Validate that all required artefacts exist (fail fast for shipping / CI / smoke tests).

Run:
  python -m src.job_intel.pipelines.ch5_app_build

Notes:
- This script does NOT launch Streamlit.
- It also does NOT retrain any models.
- It only builds the fairness assets (because we have a builder for those).
  Other artefacts (SHAP .npz, skill value index, skill similarity edges) are validated here
  but are expected to already exist from upstream pipelines/exports.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.job_intel.config import PROCESSED_DATA_DIR
from src.job_intel.pipelines.ch5_build_fairness_assets import build_fairness_assets


@dataclass(frozen=True)
class BuildReport:
    built: dict[str, Path]
    present: dict[str, Path]
    missing: dict[str, Path]


def _repo_root() -> Path:
    # .../src/job_intel/pipelines/ch5_app_build.py -> repo root = parents[3]
    return Path(__file__).resolve().parents[3]


def _manifest() -> dict[str, Path]:
    """
    Canonical artefact locations expected by the Chapter 5 app.
    Keep this aligned with the loaders in:
      - src/job_intel/app/landscape.py
      - src/job_intel/app/upskilling_macro.py
      - src/job_intel/app/recommender.py (demo persona)
    """
    repo = _repo_root()
    ch5_assets = PROCESSED_DATA_DIR / "ch5_assets"

    return {
        # demo config (optional for manual runs; required for demo button)
        "demo_persona_cfg": repo
        / "src"
        / "job_intel"
        / "evaluation"
        / "recommender_demo.json",
        # fairness / residuals
        "df_with_residuals": PROCESSED_DATA_DIR / "df_with_residuals.csv",
        "fairness_group_summary_long": ch5_assets / "fairness_group_summary_long.csv",
        "fairness_residual_box_stats": ch5_assets / "fairness_residual_box_stats.json",
        # optional (not required by UI; kept for reproducibility)
        "fairness_residual_hist_bins": ch5_assets / "fairness_residual_hist_bins.csv",
        # landscape: skill value + SHAP explanation bundle
        "skill_value_index": ch5_assets / "skill_value_index.csv",
        "shap_salary_explanation": ch5_assets / "shap_salary_explanation.npz",
        # macro co-learning (chapter 2)
        "skill_similarity_edges": PROCESSED_DATA_DIR
        / "skill_similarity_matrix"
        / "skill_similarity_edges_k5_embeddings.csv",
    }


def _validate_manifest(
    manifest: dict[str, Path], *, required_keys: set[str]
) -> tuple[dict[str, Path], dict[str, Path]]:
    present: dict[str, Path] = {}
    missing: dict[str, Path] = {}

    for k, p in manifest.items():
        if k not in required_keys:
            continue
        if p.exists():
            present[k] = p
        else:
            missing[k] = p
    return present, missing


def _print_manifest_status(
    *, built: dict[str, Path], present: dict[str, Path], missing: dict[str, Path]
) -> None:
    if built:
        print("\nBuilt:")
        for k, p in built.items():
            print(f"  - {k}: {p}")

    if present:
        print("\nPresent:")
        for k, p in present.items():
            print(f"  - {k}: {p}")

    if missing:
        print("\nMissing:")
        for k, p in missing.items():
            print(f"  - {k}: {p}")


def build_ch5_assets(
    *,
    rebuild_fairness: bool = True,
    bins: int = 30,
    strict: bool = True,
    validate_demo_cfg: bool = True,
) -> BuildReport:
    """
    Build the deterministic Chapter 5 assets and validate manifest.

    strict=True:
      - exits non-zero if any required artefact is missing.

    validate_demo_cfg:
      - if True, demo persona config is treated as required (so demo button always works).
      - if False, demo config is treated as optional (manual mode still works).
    """
    manifest = _manifest()

    built: dict[str, Path] = {}

    # 1) Build fairness assets (only artefact builder currently available)
    if rebuild_fairness:
        built = build_fairness_assets(bins=bins)

    # 2) Validate required artefacts for v1 app run
    required_keys = {
        "df_with_residuals",
        "fairness_group_summary_long",
        "fairness_residual_box_stats",
        "skill_value_index",
        "shap_salary_explanation",
        "skill_similarity_edges",
    }
    if validate_demo_cfg:
        required_keys.add("demo_persona_cfg")

    present, missing = _validate_manifest(manifest, required_keys=required_keys)

    # 3) Report + fail fast
    _print_manifest_status(built=built, present=present, missing=missing)

    if missing and strict:
        # clear actionable error
        keys = ", ".join(sorted(missing.keys()))
        raise SystemExit(
            f"\nChapter 5 build failed: missing required artefacts: {keys}\n"
            f"Fix: generate/persist the missing files (or run upstream pipelines) and re-run this builder."
        )

    return BuildReport(built=built, present=present, missing=missing)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build + validate Chapter 5 app assets")
    p.add_argument(
        "--no-rebuild-fairness",
        action="store_true",
        help="Skip rebuilding fairness assets (only validate files).",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Histogram bins for fairness residual hist bins (if rebuilding fairness).",
    )
    p.add_argument(
        "--non-strict",
        action="store_true",
        help="Do not fail on missing artefacts (still prints missing list).",
    )
    p.add_argument(
        "--no-demo-required",
        action="store_true",
        help="Treat demo persona config as optional (manual mode only).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    argv = argv if argv is not None else sys.argv[1:]
    args = _parse_args(argv)

    report = build_ch5_assets(
        rebuild_fairness=not args.no_rebuild_fairness,
        bins=int(args.bins),
        strict=not args.non_strict,
        validate_demo_cfg=not args.no_demo_required,
    )

    # return a small json-like payload for programmatic use
    return {
        "built": {k: str(v) for k, v in report.built.items()},
        "present": {k: str(v) for k, v in report.present.items()},
        "missing": {k: str(v) for k, v in report.missing.items()},
    }


if __name__ == "__main__":
    main()
