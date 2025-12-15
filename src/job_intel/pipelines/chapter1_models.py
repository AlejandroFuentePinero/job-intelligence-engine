# src/job_intel/pipelines/chapter1_models.py

"""
Chapter 1 pipeline orchestrator.

This wrapper runs the Chapter 1 model pipelines (salary and/or skill) using a
single entrypoint and returns a structured output dict for reproducibility.

Notes
-----
- This function does not build models itself; it delegates to the underlying
  pipelines.
- It enforces a minimal guardrail: at least one of salary/skill must be True.
- It runs salary first if both are requested (safe default; avoids subtle
  dependency issues if any artefacts are shared).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.job_intel.pipelines.salary_model_pipeline import run_salary_pipeline
from src.job_intel.pipelines.skill_model_pipeline import run_skill_pipeline


def run_chapter1_models(
    salary: bool = False,
    skill: bool = False,
    do_eval: bool = False,
    show_plots_eval: bool = False,
    save_model: bool = False,
) -> Dict[str, Optional[Any]]:
    """
    Run Chapter 1 pipelines (salary and/or skill) and return their outputs.

    Parameters
    ----------
    salary : bool
        If True, run the salary modelling pipeline.
    skill : bool
        If True, run the skill-requirement modelling pipeline.
    do_eval : bool
        If True, run evaluation steps (where supported by the pipeline).
    show_plots_eval : bool
        If True, display evaluation plots (where supported).
    save_model : bool
        If True, persist fitted artefacts (models, matrices, etc.) to disk.

    Returns
    -------
    dict
        {"salary": <salary_pipeline_output_or_None>, "skill": <skill_pipeline_output_or_None>}
    """
    if not salary and not skill:
        raise ValueError("Select at least one of salary=True or skill=True.")

    out: Dict[str, Optional[Any]] = {"salary": None, "skill": None}

    # Run salary first (safe default ordering).
    if salary:
        out["salary"] = run_salary_pipeline(
            do_eval=do_eval,
            show_plots_eval=show_plots_eval,
            save_model=save_model,
        )

    if skill:
        out["skill"] = run_skill_pipeline(
            show_plots_eval=show_plots_eval,
            save_model=save_model,
        )

    return out
