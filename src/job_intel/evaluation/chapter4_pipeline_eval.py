# src/job_intel/evaluation/chapter4_pipeline_eval.py
"""
Chapter 4 — Pipeline Integration Evaluation (v1)

Goal: orchestration-level checks only.
- Each Chapter 4 module already has strong internal guards.
- This script checks cross-module contracts + invariants + minimal smoke variants.

Key robustness features:
- Treats job_id as either a column or index (normalises via _ensure_job_id_col()).
- Upskilling "frozen universe" invariant is checked against recommender scored_universe size
  (NOT against a baseline rowset that upskilling does not return).
- Career simulation checks are schema-robust: searches for a df with (scenario/name + job_id).
- Optional suppression of pipeline prints/logging (quiet_pipeline=True).
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.job_intel.pipelines.chapter4_recommender import run_recommender_pipeline
from src.job_intel.v2_updates.features.career_simulator import (
    SimulationConfig,
    SimulationScenario,
)


# -----------------------------
# Path helpers
# -----------------------------
def _find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path().resolve()).resolve()
    for p in [start] + list(start.parents):
        if (p / "src").exists():
            return p
    raise FileNotFoundError(f"Could not find repo root above: {start}")


def _default_demo_path() -> Path:
    repo = _find_repo_root()
    return repo / "src" / "job_intel" / "evaluation" / "recommender_demo.json"


# -----------------------------
# Output suppression (no pipeline edits)
# -----------------------------
@contextmanager
def _quiet_context(enabled: bool = True):
    """
    Best-effort suppression of pipeline chatter without touching pipeline code:
    - Redirect stdout/stderr (captures print()).
    - Temporarily raise logging levels to WARNING for src/job_intel loggers.

    Note: if the caller configured logging handlers externally, this still works for most cases.
    """
    if not enabled:
        yield
        return

    buf_out, buf_err = io.StringIO(), io.StringIO()

    # logging level overrides (best effort)
    targets = [
        "",  # root
        "src",
        "src.job_intel",
        "src.job_intel.pipelines",
        "src.job_intel.pipelines.chapter4_recommender",
    ]
    prev_levels: dict[str, int] = {}
    for name in targets:
        lg = logging.getLogger(name)
        prev_levels[name] = lg.level
        lg.setLevel(logging.WARNING)

    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            yield
    finally:
        for name, lvl in prev_levels.items():
            logging.getLogger(name).setLevel(lvl)


# -----------------------------
# Report helpers
# -----------------------------
def _pf(cond: bool) -> str:
    return "PASS" if bool(cond) else "FAIL"


def _safe_get(d: dict[str, Any], path: list[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _require_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    missing = [c for c in cols if c not in df.columns]
    return len(missing) == 0


def _safe_set(x: Any) -> set:
    return set(pd.Series(x).dropna().tolist())


def _is_sorted_desc(s: pd.Series) -> bool:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return True
    return s.is_monotonic_decreasing


def _add(report: dict[str, str], name: str, cond: bool) -> None:
    report[name] = _pf(cond)


def _add_fail(report: dict[str, str], name: str) -> None:
    report[name] = "FAIL"


def _ensure_job_id_col(df: pd.DataFrame, *, where: str) -> pd.DataFrame:
    """
    Ensure a DataFrame has a 'job_id' column.

    Many tables use job_id as index. This normalises:
    - if 'job_id' exists -> reset_index (keeps it)
    - else -> reset_index and rename common index col -> 'job_id'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{where} must be a DataFrame")

    out = df.reset_index(drop=False).copy()

    if "job_id" in out.columns:
        return out

    if "index" in out.columns:
        return out.rename(columns={"index": "job_id"})

    idx_name = df.index.name
    if idx_name and idx_name in out.columns:
        return out.rename(columns={idx_name: "job_id"})

    out["job_id"] = df.index.astype(object)
    return out


def _first_df_with_cols(obj: Any, cols: set[str]) -> pd.DataFrame | None:
    if isinstance(obj, pd.DataFrame):
        return obj if cols.issubset(set(obj.columns)) else None
    if isinstance(obj, dict):
        for v in obj.values():
            out = _first_df_with_cols(v, cols)
            if out is not None:
                return out
    return None


def _pick_scenario_col(df: pd.DataFrame) -> str | None:
    for c in ["scenario", "name", "scenario_name"]:
        if c in df.columns:
            return c
    return None


# -----------------------------
# Demo config loader
# -----------------------------
def load_demo_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with open(path, "r") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise TypeError("Demo JSON must load to a dict.")

    for k in ["user_inputs", "pipeline_params", "career_simulation"]:
        if k not in cfg:
            raise KeyError(f"Demo JSON missing required top-level key: {k}")

    return cfg


def build_sim_objects(
    cfg: dict[str, Any],
) -> tuple[bool, list[SimulationScenario], SimulationConfig]:
    sim_block = cfg.get("career_simulation", {}) or {}
    run_career_sim = bool(sim_block.get("run_career_sim", False))

    scenarios_raw = sim_block.get("scenarios", []) or []
    scenarios = [SimulationScenario(**d) for d in scenarios_raw]

    config_raw = sim_block.get("config", {}) or {}
    sim_config = SimulationConfig(**config_raw) if config_raw else SimulationConfig()

    return run_career_sim, scenarios, sim_config


# -----------------------------
# Main evaluation
# -----------------------------
def run_ch4_pipeline_eval(
    demo_json_path: str | Path | None = None,
    *,
    run_smoke: bool = True,
    verbose: bool = True,
    quiet_pipeline: bool = True,
) -> tuple[pd.DataFrame, bool, dict[str, Any]]:
    report: dict[str, str] = {}

    demo_path = (
        Path(demo_json_path) if demo_json_path is not None else _default_demo_path()
    )

    # A) Load config + build dataclasses
    try:
        cfg = load_demo_config(demo_path)
        _add(report, "A_demo_json_loaded", True)
    except Exception as e:
        _add_fail(report, f"A_demo_json_loaded ({type(e).__name__})")
        summary = pd.DataFrame(
            {"check": list(report.keys()), "status": list(report.values())}
        )
        summary["pass_"] = summary["status"].eq("PASS")
        return summary, False, {"error": str(e), "demo_path": str(demo_path)}

    try:
        run_career_sim, scenarios, sim_config = build_sim_objects(cfg)
        _add(report, "A_sim_dataclasses_built", True)
    except Exception as e:
        _add_fail(report, f"A_sim_dataclasses_built ({type(e).__name__})")
        summary = pd.DataFrame(
            {"check": list(report.keys()), "status": list(report.values())}
        )
        summary["pass_"] = summary["status"].eq("PASS")
        return (
            summary,
            False,
            {"error": str(e), "demo_path": str(demo_path), "cfg": cfg},
        )

    user_inputs = cfg["user_inputs"]
    pipeline_params = cfg.get("pipeline_params", {}) or {}

    # Force low-noise settings without touching pipeline code.
    # These keys exist in your pipeline signature and downstream modules.
    pipeline_params = dict(pipeline_params)
    pipeline_params.setdefault("verbose", False)  # job_recommender verbosity
    pipeline_params.setdefault("print_report", False)  # upskilling report printing

    # B) Run pipeline
    try:
        with _quiet_context(quiet_pipeline):
            recommender, explanation, upskill, career_sim = run_recommender_pipeline(
                **user_inputs,
                **pipeline_params,
                run_career_sim=run_career_sim,
                scenarios=scenarios,
                config=sim_config,
            )
        _add(report, "B_pipeline_runs", True)
    except Exception as e:
        _add_fail(report, f"B_pipeline_runs ({type(e).__name__})")
        summary = pd.DataFrame(
            {"check": list(report.keys()), "status": list(report.values())}
        )
        summary["pass_"] = summary["status"].eq("PASS")
        return (
            summary,
            False,
            {"error": str(e), "demo_path": str(demo_path), "cfg": cfg},
        )

    # Pre-normalise common tables
    rec_uni_raw = _safe_get(recommender, ["tables", "scored_universe"])
    rec_uni = (
        _ensure_job_id_col(rec_uni_raw, where="recommender.tables.scored_universe")
        if isinstance(rec_uni_raw, pd.DataFrame)
        else None
    )
    expected_universe_n = (
        int(rec_uni["job_id"].nunique()) if isinstance(rec_uni, pd.DataFrame) else None
    )

    # -----------------------------
    # C) Recommender checks
    # -----------------------------
    _add(
        report,
        "Recommender_tables",
        isinstance(_safe_get(recommender, ["tables"]), dict),
    )
    _add(report, "Recommender_tables_universe", isinstance(rec_uni_raw, pd.DataFrame))
    _add(
        report,
        "Recommender_tables_best_now",
        isinstance(_safe_get(recommender, ["tables", "top_best_now"]), pd.DataFrame),
    )
    _add(
        report,
        "Recommender_tables_stretch",
        isinstance(_safe_get(recommender, ["tables", "top_stretch"]), pd.DataFrame),
    )

    required_cols = [
        "job_id",
        "score",
        "bucket",
        "suitability",
        "competitiveness_index",
    ]

    top_best_raw = _safe_get(recommender, ["tables", "top_best_now"])
    top_str_raw = _safe_get(recommender, ["tables", "top_stretch"])
    top_best = (
        _ensure_job_id_col(top_best_raw, where="recommender.tables.top_best_now")
        if isinstance(top_best_raw, pd.DataFrame)
        else None
    )
    top_str = (
        _ensure_job_id_col(top_str_raw, where="recommender.tables.top_stretch")
        if isinstance(top_str_raw, pd.DataFrame)
        else None
    )

    if isinstance(top_str, pd.DataFrame):
        _add(
            report,
            "Recommender_tables_stretch_cols",
            _require_cols(top_str, required_cols),
        )
        _add(
            report,
            "Recommender_tables_stretch_job_id_nonnull",
            ~top_str["job_id"].isna().any(),
        )
        _add(
            report,
            "Recommender_tables_stretch_score_nonnull",
            ~top_str["score"].isna().any(),
        )
        _add(
            report,
            "Recommender_tables_stretch_bucket_valid",
            ~top_str["bucket"].eq("best_now").any(),
        )
    else:
        _add_fail(report, "Recommender_tables_stretch_cols")
        _add_fail(report, "Recommender_tables_stretch_job_id_nonnull")
        _add_fail(report, "Recommender_tables_stretch_score_nonnull")
        _add_fail(report, "Recommender_tables_stretch_bucket_valid")

    if isinstance(top_best, pd.DataFrame):
        _add(
            report,
            "Recommender_tables_best_now_cols",
            _require_cols(top_best, required_cols),
        )
        _add(
            report,
            "Recommender_tables_best_now_job_id_nonnull",
            ~top_best["job_id"].isna().any(),
        )
        _add(
            report,
            "Recommender_tables_best_now_score_nonnull",
            ~top_best["score"].isna().any(),
        )
        _add(
            report,
            "Recommender_tables_best_now_bucket_valid",
            ~top_best["bucket"].eq("stretch").any(),
        )
    else:
        _add_fail(report, "Recommender_tables_best_now_cols")
        _add_fail(report, "Recommender_tables_best_now_job_id_nonnull")
        _add_fail(report, "Recommender_tables_best_now_score_nonnull")
        _add_fail(report, "Recommender_tables_best_now_bucket_valid")

    # Bucket invariants
    if isinstance(top_best, pd.DataFrame) and isinstance(top_str, pd.DataFrame):
        params = recommender.get("params", {}) if isinstance(recommender, dict) else {}
        top_n_best = params.get("top_n_best", None)
        top_n_stretch = params.get("top_n_stretch", None)

        _add(
            report,
            "Recommender_best_now_bucket_size",
            top_n_best is not None and len(top_best) == int(top_n_best),
        )
        _add(
            report,
            "Recommender_stretch_bucket_size",
            top_n_stretch is not None and len(top_str) == int(top_n_stretch),
        )

        best_ids = _safe_set(top_best["job_id"])
        stretch_ids = _safe_set(top_str["job_id"])
        _add(
            report, "Recommender_bucket_overlap_empty", len(best_ids & stretch_ids) == 0
        )

        _add(
            report,
            "Recommender_best_now_score_sorted",
            _is_sorted_desc(top_best["score"]),
        )
        _add(
            report,
            "Recommender_stretch_score_sorted",
            _is_sorted_desc(top_str["score"]),
        )
    else:
        _add_fail(report, "Recommender_best_now_bucket_size")
        _add_fail(report, "Recommender_stretch_bucket_size")
        _add_fail(report, "Recommender_bucket_overlap_empty")
        _add_fail(report, "Recommender_best_now_score_sorted")
        _add_fail(report, "Recommender_stretch_score_sorted")

    # -----------------------------
    # D) Explanation checks
    # -----------------------------
    _add(
        report,
        "Explanation_tables",
        isinstance(_safe_get(explanation, ["tables"]), dict),
    )
    exp_uni_raw = _safe_get(explanation, ["tables", "scored_universe_explained"])
    _add(report, "Explanation_tables_universe", isinstance(exp_uni_raw, pd.DataFrame))

    exp_uni = (
        _ensure_job_id_col(
            exp_uni_raw, where="explanation.tables.scored_universe_explained"
        )
        if isinstance(exp_uni_raw, pd.DataFrame)
        else None
    )
    top_best_exp_raw = _safe_get(explanation, ["tables", "top_best_explained"])
    top_str_exp_raw = _safe_get(explanation, ["tables", "top_stretch_explained"])
    top_best_exp = (
        _ensure_job_id_col(
            top_best_exp_raw, where="explanation.tables.top_best_explained"
        )
        if isinstance(top_best_exp_raw, pd.DataFrame)
        else None
    )
    top_str_exp = (
        _ensure_job_id_col(
            top_str_exp_raw, where="explanation.tables.top_stretch_explained"
        )
        if isinstance(top_str_exp_raw, pd.DataFrame)
        else None
    )

    required_cols_exp = [
        "job_id",
        "missing_families",
        "n_missing_families",
        "salary_context",
        "why_rank",
        "covered_families",
        "why_bucket",
        "n_covered_families",
    ]

    if isinstance(top_best_exp, pd.DataFrame):
        _add(
            report,
            "Explanation_best_now_cols",
            _require_cols(top_best_exp, required_cols_exp),
        )
        _add(
            report,
            "Explanation_best_now_missing_type",
            isinstance(top_best_exp["missing_families"].iloc[0], list),
        )
    else:
        _add_fail(report, "Explanation_best_now_cols")
        _add_fail(report, "Explanation_best_now_missing_type")

    if isinstance(top_str_exp, pd.DataFrame):
        _add(
            report,
            "Explanation_stretch_cols",
            _require_cols(top_str_exp, required_cols_exp),
        )
        _add(
            report,
            "Explanation_stretch_missing_type",
            isinstance(top_str_exp["missing_families"].iloc[0], list),
        )
    else:
        _add_fail(report, "Explanation_stretch_cols")
        _add_fail(report, "Explanation_stretch_missing_type")

    if isinstance(exp_uni, pd.DataFrame) and isinstance(top_best_exp, pd.DataFrame):
        uni_set = _safe_set(exp_uni["job_id"])
        _add(
            report,
            "Explanation_best_now_ids_in_universe",
            _safe_set(top_best_exp["job_id"]).issubset(uni_set),
        )
    else:
        _add_fail(report, "Explanation_best_now_ids_in_universe")

    if isinstance(exp_uni, pd.DataFrame) and isinstance(top_str_exp, pd.DataFrame):
        uni_set = _safe_set(exp_uni["job_id"])
        _add(
            report,
            "Explanation_stretch_ids_in_universe",
            _safe_set(top_str_exp["job_id"]).issubset(uni_set),
        )
    else:
        _add_fail(report, "Explanation_stretch_ids_in_universe")

    # -----------------------------
    # E) Upskilling checks
    # -----------------------------
    _add(report, "Upskill_is_dict", isinstance(upskill, dict))
    _add(
        report,
        "Upskill_recommendation_df",
        isinstance(upskill.get("upskill_recommendation", None), pd.DataFrame),
    )

    if isinstance(upskill.get("upskill_recommendation", None), pd.DataFrame):
        expected_topn = int(_safe_get(upskill, ["meta", "top_n_skills"]) or 0)
        _add(
            report,
            "Upskill_recommendation_topN",
            len(upskill["upskill_recommendation"]) == expected_topn,
        )
    else:
        _add_fail(report, "Upskill_recommendation_topN")

    _add(
        report,
        "Upskill_recommendation_dict",
        isinstance(upskill.get("recommendation_dict", None), dict),
    )

    scenario_meta = upskill.get("scenario_meta", None)
    if isinstance(scenario_meta, pd.DataFrame) and "status" in scenario_meta.columns:
        _add(
            report,
            "Upskill_scenario_meta_status",
            scenario_meta["status"].isin(["kept", "skipped"]).all(),
        )
    else:
        _add_fail(report, "Upskill_scenario_meta_status")

    # FIX #1: Frozen universe invariant must be checked against recommender scored_universe size
    # (upskilling does not return baseline rows in job_base_upskill).
    base_df = upskill.get("job_base_upskill", None)
    if (
        isinstance(base_df, pd.DataFrame)
        and {"upskill_scenario", "job_id"}.issubset(base_df.columns)
        and expected_universe_n is not None
    ):
        per_s = base_df.groupby("upskill_scenario")["job_id"].nunique()
        _add(
            report,
            "Upskill_frozen_universe_invariant",
            (per_s == expected_universe_n).all(),
        )
    else:
        _add_fail(report, "Upskill_frozen_universe_invariant")

    upsum = upskill.get("upskill_summary", None)
    if isinstance(upsum, pd.DataFrame) and "passes_guardrail" in upsum.columns:
        _add(
            report,
            "Upskill_guardrail_all_pass",
            bool(upsum["passes_guardrail"].fillna(False).all()),
        )
    else:
        _add_fail(report, "Upskill_guardrail_all_pass")

    # -----------------------------
    # F) Career simulation checks (schema-robust)
    # -----------------------------
    if run_career_sim:
        _add(report, "CareerSim_exists", isinstance(career_sim, dict))

        # FIX #2: locate any df with (scenario/name/scenario_name + job_id)
        df_sj = _first_df_with_cols(career_sim, {"job_id", "scenario"})
        if df_sj is None:
            df_sj = _first_df_with_cols(career_sim, {"job_id", "name"})
        if df_sj is None:
            df_sj = _first_df_with_cols(career_sim, {"job_id", "scenario_name"})

        if isinstance(df_sj, pd.DataFrame):
            scen_col = _pick_scenario_col(df_sj)
            n_out = int(df_sj[scen_col].nunique()) if scen_col is not None else None
            _add(
                report,
                "CareerSim_scenario_count_matches",
                (n_out == len(scenarios)) if n_out is not None else False,
            )

            # FIX #3: frozen-universe check when required
            if (
                getattr(sim_config, "require_frozen_universe", False)
                and scen_col is not None
            ):
                per_s = df_sj.groupby(scen_col)["job_id"].nunique()
                frozen_ok = bool((per_s == per_s.iloc[0]).all())
                _add(report, "CareerSim_frozen_universe_or_subset", frozen_ok)
            else:
                report["CareerSim_frozen_universe_or_subset"] = "PASS"
        else:
            _add_fail(report, "CareerSim_scenario_count_matches")
            _add_fail(report, "CareerSim_frozen_universe_or_subset")

        # outputs present (any dataframe anywhere)
        def _has_df(o: Any) -> bool:
            if isinstance(o, pd.DataFrame):
                return True
            if isinstance(o, dict):
                return any(_has_df(v) for v in o.values())
            return False

        _add(
            report,
            "CareerSim_outputs_present",
            isinstance(career_sim, dict) and _has_df(career_sim),
        )

    else:
        report["CareerSim_exists"] = "PASS"
        report["CareerSim_outputs_present"] = "PASS"
        report["CareerSim_scenario_count_matches"] = "PASS"
        report["CareerSim_frozen_universe_or_subset"] = "PASS"

    # -----------------------------
    # G) Cross-module consistency
    # -----------------------------
    if (
        isinstance(exp_uni, pd.DataFrame)
        and isinstance(top_best, pd.DataFrame)
        and isinstance(top_str, pd.DataFrame)
    ):
        exp_set = _safe_set(exp_uni["job_id"])
        _add(
            report,
            "Cross_best_now_ids_in_explained_universe",
            _safe_set(top_best["job_id"]).issubset(exp_set),
        )
        _add(
            report,
            "Cross_stretch_ids_in_explained_universe",
            _safe_set(top_str["job_id"]).issubset(exp_set),
        )
    else:
        _add_fail(report, "Cross_best_now_ids_in_explained_universe")
        _add_fail(report, "Cross_stretch_ids_in_explained_universe")

    if isinstance(upskill.get("upskill_recommendation", None), pd.DataFrame):
        scen_idx = set(upskill["upskill_recommendation"].index.tolist())
        _add(
            report,
            "Cross_upskill_recommendation_not_baseline",
            "baseline" not in scen_idx,
        )
    else:
        _add_fail(report, "Cross_upskill_recommendation_not_baseline")

    # -----------------------------
    # H) Smoke variants (optional)
    # -----------------------------
    def _run_smoke(
        name: str,
        *,
        expected_exceptions: tuple[type[Exception], ...] = (),
        **kwargs: Any,
    ) -> None:
        try:
            with _quiet_context(quiet_pipeline):
                _ = run_recommender_pipeline(**kwargs)
            report[name] = "PASS"
        except expected_exceptions:
            report[name] = "PASS"
        except Exception as e:
            report[name] = f"FAIL: {type(e).__name__}: {str(e)[:120]}"

    if run_smoke:
        base_kwargs = dict(
            **user_inputs,
            **pipeline_params,
            run_career_sim=run_career_sim,
            scenarios=scenarios,
            config=sim_config,
        )

        # This may legitimately raise ValueError if stretch bucket cannot be formed.
        kw = base_kwargs.copy()
        kw["skill_text"] = ""
        _run_smoke("Smoke_empty_skill_text", expected_exceptions=(ValueError,), **kw)

        kw = base_kwargs.copy()
        kw["salary_target"] = None
        _run_smoke("Smoke_salary_target_none", **kw)

        kw = base_kwargs.copy()
        kw["run_explanator"] = False
        kw["run_upskilling"] = False
        kw["run_career_sim"] = False
        kw["scenarios"] = None
        kw["config"] = SimulationConfig()
        _run_smoke("Smoke_baseline_only", **kw)

    # -----------------------------
    # Summarize
    # -----------------------------
    summary = pd.DataFrame(
        {"check": list(report.keys()), "status": list(report.values())}
    )
    summary["pass_"] = summary["status"].eq("PASS")
    all_passed = bool(summary["pass_"].all())

    bundle = {
        "demo_path": str(demo_path),
        "cfg": cfg,
        "resolved": {
            "run_career_sim": run_career_sim,
            "scenarios": [asdict(s) for s in scenarios],
            "config": asdict(sim_config),
        },
        "outputs": {
            "recommender": recommender,
            "explanation": explanation,
            "upskill": upskill,
            "career_sim": career_sim,
        },
    }

    if verbose:
        failed = summary.loc[~summary["pass_"]]
        print(f"Chapter 4 pipeline eval: {'PASS' if all_passed else 'FAIL'}")
        if len(failed) > 0:
            print("\nFailed checks:")
            print(failed[["check", "status"]].to_string(index=False))

    return summary, all_passed, bundle


def main() -> None:
    summary, all_passed, _ = run_ch4_pipeline_eval(
        run_smoke=True, verbose=True, quiet_pipeline=True
    )
    print("\nSummary:")
    print(summary.to_string(index=False))
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
