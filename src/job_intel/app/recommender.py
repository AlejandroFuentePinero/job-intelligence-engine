# src/job_intel/app/recommender.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.job_intel.pipelines.chapter4_recommender import run_recommender_pipeline

# Only used as a default config object (we do NOT run career simulation)
from src.job_intel.v2_updates.features.career_simulator import SimulationConfig


@dataclass(frozen=True)
class AppResult:
    narrative: str
    payload: dict[str, Any]


def _repo_root() -> Path:
    # .../src/job_intel/app/recommender.py -> repo root = parents[3]
    return Path(__file__).resolve().parents[3]


def _default_demo_path() -> Path:
    return _repo_root() / "src" / "job_intel" / "evaluation" / "recommender_demo.json"


def _ensure_job_id(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    if "job_id" in df.columns:
        return df
    if df.index.name == "job_id":
        return df.reset_index()
    out = df.reset_index()
    if "index" in out.columns and "job_id" not in out.columns:
        out = out.rename(columns={"index": "job_id"})
    return out


def _pick_desc_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    if "Job Description" in df.columns:
        return "Job Description"
    for c in ["job_description", "description", "desc"]:
        if c in df.columns:
            return c
    return None


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "job_id",
        "title_rich",
        "state",
        "Sector",
        "Industry",
        "score",
        "suitability",
        "competitiveness_index",
        "pred_sal",
        "sal_mean",
        "n_missing_families",
        "n_covered_families",
        "why_bucket",
        "why_rank",
        "salary_context",
    ]
    cols = [c for c in wanted if c in df.columns]
    return df[cols] if cols else df


def _split_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Accepts either:
      - demo_cfg dict: {"user_inputs": {...}, "pipeline_params": {...}, ...}
      - manual profile dict: {"skill_text":..., "current_state":..., ...}

    Returns (user_inputs, pipeline_params)
    """
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be dict, got {type(payload)}")

    if "user_inputs" in payload:
        user_inputs = payload.get("user_inputs") or {}
        pipeline_params = payload.get("pipeline_params") or {}
        if not isinstance(user_inputs, dict) or not isinstance(pipeline_params, dict):
            raise TypeError("demo_cfg user_inputs/pipeline_params must be dicts")
        return user_inputs, pipeline_params

    # manual mode (already user_inputs)
    return payload, {}


def _series_summary(x: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(x, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    return {
        "n": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        "p10": float(q.get(0.1, np.nan)),
        "q1": float(q.get(0.25, np.nan)),
        "median": float(q.get(0.5, np.nan)),
        "q3": float(q.get(0.75, np.nan)),
        "p90": float(q.get(0.9, np.nan)),
    }


def _candidate_positioning_summary(candidate_jobs: pd.DataFrame) -> dict[str, Any]:
    if candidate_jobs is None or not isinstance(candidate_jobs, pd.DataFrame):
        return {}

    out: dict[str, Any] = {"n_candidates": int(candidate_jobs.shape[0])}

    for col in [
        "suitability",
        "competitiveness_index",
        "pred_sal",
        "sal_mean",
        "score",
    ]:
        if col in candidate_jobs.columns:
            out[col] = _series_summary(candidate_jobs[col])

    # light categorical context (only if present)
    for col in ["state", "Sector", "Industry", "title_rich"]:
        if col in candidate_jobs.columns:
            vc = (
                candidate_jobs[col]
                .astype(str)
                .replace("nan", np.nan)
                .dropna()
                .value_counts()
                .head(8)
            )
            if not vc.empty:
                out[f"top_{col}"] = vc.to_dict()

    return out


def _run_ch4(payload: dict[str, Any]) -> tuple[AppResult, dict[str, Any]]:
    """
    Runs the Chapter 4 pipeline once, storing:
      - AppResult for the Recommender page display
      - ch4 bundle (rec/expl/up) for other pages (Upskilling + Macro etc.)

    IMPORTANT: career simulation is ALWAYS disabled here.
    """
    user_inputs, pipeline_params = _split_payload(payload)

    # Copy then FORCE app defaults (fast + consistent)
    pipeline_params = dict(pipeline_params)

    # CRITICAL: remove any sim keys so we don't pass duplicates
    pipeline_params.pop("run_career_sim", None)
    pipeline_params.pop("scenarios", None)
    pipeline_params.pop("config", None)

    # Force app toggles (deterministic, fast)
    pipeline_params["run_explanator"] = True
    pipeline_params["run_upskilling"] = True
    pipeline_params["print_report"] = False
    pipeline_params["verbose"] = False
    # If your pipeline supports it, keep it light:
    pipeline_params["include_scored_universe"] = False

    # Run pipeline (career sim explicitly disabled)
    rec_out, expl_out, up_out, sim_out = run_recommender_pipeline(
        **user_inputs,
        **pipeline_params,
        run_career_sim=False,
        scenarios=None,
        config=SimulationConfig(),
    )

    # Build explained tables payload
    best_df = None
    stretch_df = None
    glossary: dict[str, Any] = {}

    if isinstance(expl_out, dict):
        tables = expl_out.get("tables", {}) or {}
        best_df = tables.get("top_best_explained")
        stretch_df = tables.get("top_stretch_explained")
        glossary = expl_out.get("metric_glossary", {}) or {}

    # Fallback: if explanations failed, use raw top tables
    if best_df is None and isinstance(rec_out, dict):
        best_df = rec_out.get("tables", {}).get("top_best_now")
    if stretch_df is None and isinstance(rec_out, dict):
        stretch_df = rec_out.get("tables", {}).get("top_stretch")

    candidate_jobs = None
    counts: dict[str, Any] = {}
    salary_summary: dict[str, Any] = {}
    warnings: list[Any] = []

    if isinstance(rec_out, dict):
        tables = rec_out.get("tables", {}) or {}
        candidate_jobs = tables.get("candidate_jobs")
        counts = rec_out.get("counts", {}) or {}
        salary_summary = rec_out.get("salary_summary", {}) or {}
        warnings = rec_out.get("warnings", []) or []

    # Narrative (minimal)
    n_best = int(best_df.shape[0]) if isinstance(best_df, pd.DataFrame) else 0
    n_stretch = int(stretch_df.shape[0]) if isinstance(stretch_df, pd.DataFrame) else 0
    state = str(user_inputs.get("current_state", "ALL") or "ALL")
    salary_target = user_inputs.get("salary_target", None)

    narrative = (
        f"Computed recommendations under current constraints (state={state}"
        + (f", salary_target={salary_target}" if salary_target else "")
        + f"). Returning {n_best} best_now and {n_stretch} stretch roles."
    )

    app_payload = {
        "best_now_explained": best_df,
        "stretch_explained": stretch_df,
        "candidate_jobs": candidate_jobs,
        "metric_glossary": glossary,
        "counts": counts,
        "salary_summary": salary_summary,
        "warnings": warnings,
        "positioning_summary": (
            _candidate_positioning_summary(candidate_jobs)
            if isinstance(candidate_jobs, pd.DataFrame)
            else {}
        ),
    }

    res = AppResult(narrative=narrative, payload=app_payload)

    ch4_bundle = {
        "recommender_out": rec_out,
        "explanation_out": expl_out,
        "upskilling_out": up_out,
        "career_sim": None,  # explicitly disabled for app runs
    }

    return res, ch4_bundle


def render() -> None:
    st.title("Recommender")

    if "profile" not in st.session_state:
        st.session_state["profile"] = {}
    if "demo_cfg" not in st.session_state:
        st.session_state["demo_cfg"] = None
    if "result" not in st.session_state:
        st.session_state["result"] = None

    # Shared bundle for other pages
    if "ch4_results" not in st.session_state:
        st.session_state["ch4_results"] = None

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Load demo persona"):
            demo_path = _default_demo_path()
            try:
                with open(demo_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                st.session_state["demo_cfg"] = cfg

                ui = cfg.get("user_inputs", {}) or {}
                st.session_state["profile"] = {
                    "skill_text": ui.get("skill_text", "") or "",
                    "current_state": ui.get("current_state", "ALL") or "ALL",
                    "job_title_family": ui.get("job_title_family", None),
                    "job_title_rich": ui.get("job_title_rich", None),
                    "target_sectors": ui.get("target_sectors", None),
                    "salary_target": ui.get("salary_target", None),
                }
                st.success("Demo loaded.")
            except Exception as e:
                st.error(f"Failed to load demo: {e}")

    with col2:
        if st.button("Reset"):
            st.session_state["profile"] = {}
            st.session_state["demo_cfg"] = None
            st.session_state["result"] = None
            st.session_state["ch4_results"] = None
            # convenience keys (optional)
            st.session_state.pop("recommender_out", None)
            st.session_state.pop("explanation_out", None)
            st.session_state.pop("upskilling_out", None)

    with col3:
        if st.button("Run recommender", type="primary"):
            payload = st.session_state["demo_cfg"] or st.session_state["profile"]
            try:
                with st.spinner(
                    "Running pipeline (recommender + explanations + upskilling)…"
                ):
                    res, bundle = _run_ch4(payload)

                st.session_state["result"] = res
                st.session_state["ch4_results"] = bundle

                # Convenience keys (optional)
                st.session_state["recommender_out"] = bundle["recommender_out"]
                st.session_state["explanation_out"] = bundle["explanation_out"]
                st.session_state["upskilling_out"] = bundle["upskilling_out"]

                st.success("Run complete.")
            except Exception as e:
                st.session_state["result"] = AppResult(
                    narrative="Pipeline error.",
                    payload={"error": str(e)},
                )
                st.session_state["ch4_results"] = None
                st.error(f"Pipeline failed: {e}")

    st.divider()
    st.subheader("User inputs")

    profile = st.session_state["profile"] or {}
    with st.form("profile_form", clear_on_submit=False):
        skill_text = st.text_area(
            "skill_text", value=profile.get("skill_text", ""), height=120
        )
        current_state = st.text_input(
            "current_state (e.g., CA or ALL)", value=profile.get("current_state", "ALL")
        )
        job_title_family = st.text_input(
            "job_title_family (optional)",
            value=profile.get("job_title_family", "") or "",
        )
        job_title_rich = st.text_input(
            "job_title_rich (optional)", value=profile.get("job_title_rich", "") or ""
        )
        target_sectors_raw = st.text_input(
            "target_sectors (optional, comma-separated)",
            value=",".join(profile.get("target_sectors", []) or []),
        )
        salary_target = st.number_input(
            "salary_target (optional; 0 = None)",
            value=float(profile.get("salary_target", 0) or 0),
            min_value=0.0,
            step=5000.0,
        )
        ok = st.form_submit_button("Update inputs")

    if ok:
        raw = (target_sectors_raw or "").replace(";", ",")
        parts = [p.strip() for p in raw.split(",")]

        target_sectors = []
        for p in parts:
            p = p.strip().strip("'").strip('"')
            if p:
                target_sectors.append(p)

        target_sectors = target_sectors or None

        st.session_state["profile"] = {
            "skill_text": skill_text,
            "current_state": current_state.strip() or "ALL",
            "job_title_family": job_title_family.strip() or None,
            "job_title_rich": job_title_rich.strip() or None,
            "target_sectors": target_sectors or None,
            "salary_target": None if salary_target == 0 else int(salary_target),
        }
        st.session_state["demo_cfg"] = None
        st.success("Inputs updated (manual mode).")

    st.divider()
    st.subheader("Results")

    res = st.session_state["result"]
    if res is None:
        st.info("Click **Run recommender**.")
        return

    if isinstance(res, dict):
        st.error("Unexpected result type (dict). Expected AppResult.")
        st.json({"result_keys": list(res.keys())})
        return

    st.write(res.narrative)

    if "error" in (res.payload or {}):
        st.error("Pipeline error")
        st.json(res.payload)
        return

    # --- positioning summary (dedicated, compact, optional) ---
    positioning = res.payload.get("positioning_summary", {}) or {}
    counts = res.payload.get("counts", {}) or {}
    salary_summary = res.payload.get("salary_summary", {}) or {}
    warnings = res.payload.get("warnings", []) or []

    if positioning or counts or salary_summary or warnings:
        with st.expander("User positioning summary", expanded=False):
            if counts:
                st.markdown("**Pipeline counts (from Chapter 4)**")
                st.json(counts)

            if salary_summary:
                st.markdown("**Salary summary (from Chapter 4)**")
                st.json(salary_summary)

            if positioning:
                st.markdown(
                    "**Candidate market under your constraints (computed from candidate_jobs)**"
                )
                st.json(positioning)

            if warnings:
                st.markdown("**Warnings**")
                st.json(warnings)

    best = _ensure_job_id(res.payload.get("best_now_explained"))
    stretch = _ensure_job_id(res.payload.get("stretch_explained"))
    candidate_jobs = res.payload.get("candidate_jobs")
    glossary = res.payload.get("metric_glossary", {})

    if not isinstance(best, pd.DataFrame) or not isinstance(stretch, pd.DataFrame):
        st.error("Missing tables in payload.")
        st.json({"payload_keys": list(res.payload.keys())})
        return

    st.markdown("## Best now")
    st.dataframe(_compact(best), use_container_width=True, hide_index=True)

    st.markdown("## Stretch")
    st.dataframe(_compact(stretch), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Job detail")

    ids = (
        pd.concat([best["job_id"], stretch["job_id"]], ignore_index=True)
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if not ids:
        st.info("No job_ids to display.")
        return

    selected = st.selectbox("Select job_id", ids)

    if isinstance(candidate_jobs, pd.DataFrame):
        cj = _ensure_job_id(candidate_jobs)
        desc_col = _pick_desc_col(cj)

        row = None
        m1 = best["job_id"].astype(str) == str(selected)
        m2 = stretch["job_id"].astype(str) == str(selected)
        if m1.any():
            row = best.loc[m1].iloc[0]
        elif m2.any():
            row = stretch.loc[m2].iloc[0]

        if row is not None:
            for k in ["why_bucket", "why_rank", "salary_context"]:
                if k in row.index:
                    st.markdown(f"**{k}**: {row[k]}")

            for k in ["covered_families", "missing_families"]:
                if k in row.index:
                    st.markdown(f"**{k}**:")
                    st.write(row[k])

        if desc_col is None:
            st.info("No job description column found (expected 'Job Description').")
        else:
            match = cj.loc[cj["job_id"].astype(str) == str(selected)]
            if len(match) == 0:
                st.info("Selected job_id not found in candidate_jobs.")
            else:
                st.markdown("### Job description")
                st.text_area("", value=str(match.iloc[0][desc_col]), height=260)
    else:
        st.info("candidate_jobs missing; cannot show job descriptions yet.")

    st.divider()
    st.subheader("Glossary")
    if isinstance(glossary, dict) and glossary:
        gloss_df = pd.DataFrame(
            {"metric": list(glossary.keys()), "meaning": list(glossary.values())}
        )
        with st.expander("Show glossary", expanded=False):
            st.dataframe(
                gloss_df.sort_values("metric"),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No glossary available.")
