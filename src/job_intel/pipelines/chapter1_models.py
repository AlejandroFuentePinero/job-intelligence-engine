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

from typing import Any, Dict, Optional, Tuple

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

    Returns
    -------
    dict
        {
          "salary": {
              "model": <XGBRegressor>,
              "metrics": <dict|None>,
              "pca": <PCA>,
              "df": <pd.DataFrame>   # includes job_id for Chapter 2 entrypoint
          } | None,
          "skill": {
              "results_df": <pd.DataFrame>,
              "prob_mat": <pd.DataFrame | np.ndarray>
          } | None
        }
    """
    if not salary and not skill:
        raise ValueError("Select at least one of salary=True or skill=True.")

    out: Dict[str, Optional[Any]] = {"salary": None, "skill": None}

    # Run salary first (safe default ordering).
    if salary:
        model, metrics, pca, df_ch1 = run_salary_pipeline(
            do_eval=do_eval,
            show_plots_eval=show_plots_eval,
            save_model=save_model,
        )

        out["salary"] = {
            "model": model,
            "metrics": metrics,
            "pca": pca,
            "df": df_ch1,
        }

    if skill:
        results_df, prob_mat = run_skill_pipeline(
            show_plots_eval=show_plots_eval,
            save_model=save_model,
        )

        out["skill"] = {
            "results_df": results_df,
            "prob_mat": prob_mat,
        }

    return out
