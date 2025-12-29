# src/job_intel/features/job_explanations.py

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _require_keys(d: dict[str, Any], keys: list[str], *, where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"{where} is missing required keys: {missing}")


def _require_cols(df: pd.DataFrame, cols: list[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where} is missing required columns: {missing}")


def _ensure_job_id_col(df: pd.DataFrame, *, where: str) -> pd.DataFrame:
    """Ensure job_id exists as a column (Top-N tables often come indexed by job_id)."""
    if "job_id" in df.columns:
        return df
    if df.index.name == "job_id":
        return df.reset_index()
    raise KeyError(
        f"{where} must contain 'job_id' either as a column or as the index name."
    )


def build_job_explanations(
    *,
    rec: dict[str, Any],
    tau: float = 0.50,
    validate: bool = True,
    include_scored_universe: bool = True,
) -> dict[str, Any]:
    """
    Chapter 4 — Job Explanations (v1)

    Augments the Chapter 4 `job_recommender()` output with human-readable explanations.

    Adds (optionally) an explained scored-universe table to support upskilling:
      - rec["tables"]["scored_universe"] must exist (from recommender drop-in)
      - output includes tables["scored_universe_explained"]

    See original docstring for full details.
    """
    if not (0.0 <= float(tau) <= 1.0):
        raise ValueError(f"tau must be in [0,1], got {tau}")

    _require_keys(rec, ["tables", "params", "profile"], where="rec")
    _require_keys(
        rec["tables"],
        ["top_best_now", "top_stretch", "candidate_jobs", "skill_prob_matrix"],
        where="rec['tables']",
    )
    if include_scored_universe:
        _require_keys(
            rec["tables"],
            ["scored_universe"],
            where="rec['tables'] (include_scored_universe=True)",
        )

    _require_keys(rec["params"], ["c_max", "alpha"], where="rec['params']")
    _require_keys(rec["profile"], ["derived"], where="rec['profile']")
    _require_keys(
        rec["profile"]["derived"], ["skill_vector"], where="rec['profile']['derived']"
    )

    params = rec["params"]
    top_best = _ensure_job_id_col(
        rec["tables"]["top_best_now"], where="rec['tables']['top_best_now']"
    ).copy()
    top_stretch = _ensure_job_id_col(
        rec["tables"]["top_stretch"], where="rec['tables']['top_stretch']"
    ).copy()
    candidate_jobs = rec["tables"]["candidate_jobs"].copy()
    skill_mat = rec["tables"]["skill_prob_matrix"].copy()
    profile = rec["profile"]

    scored_universe = None
    if include_scored_universe:
        scored_universe = _ensure_job_id_col(
            rec["tables"]["scored_universe"], where="rec['tables']['scored_universe']"
        ).copy()

    # Required columns present in Top-N tables from recommender
    base_cols = [
        "job_id",
        "suitability",
        "competitiveness_index",
        "score",
        "sal_mean",
        "pred_sal",
    ]
    _require_cols(top_best, base_cols, where="top_best")
    _require_cols(top_stretch, base_cols, where="top_stretch")

    if include_scored_universe:
        # scored_universe is for upskilling; must include bucket + score at minimum
        scored_cols = base_cols + ["bucket"]
        _require_cols(scored_universe, scored_cols, where="scored_universe")

    # -----------------------------
    # Deterministic explanation fields
    # -----------------------------
    c_max = float(params["c_max"])
    alpha = float(params["alpha"])

    def _add_bucket_rank_salary_explanations(
        df: pd.DataFrame, *, bucket_mode: str
    ) -> pd.DataFrame:
        """
        bucket_mode:
          - "best_now": assumes all rows are best_now (wording)
          - "stretch": assumes all rows are stretch
          - "mixed": uses df['bucket'] to decide wording
        """
        out = df.copy()

        if bucket_mode == "best_now":
            out["why_bucket"] = out["competitiveness_index"].map(
                lambda x: f"Barrier is low (competitiveness {x:.2f} ≤ {c_max:.2f})"
            )
        elif bucket_mode == "stretch":
            out["why_bucket"] = out["competitiveness_index"].map(
                lambda x: f"Barrier is higher (competitiveness {x:.2f} > {c_max:.2f})"
            )
        elif bucket_mode == "mixed":
            out["why_bucket"] = np.where(
                out["bucket"].astype(str) == "best_now",
                out["competitiveness_index"].map(
                    lambda x: f"Barrier is low (competitiveness {x:.2f} ≤ {c_max:.2f})"
                ),
                out["competitiveness_index"].map(
                    lambda x: f"Barrier is higher (competitiveness {x:.2f} > {c_max:.2f})"
                ),
            )
        else:
            raise ValueError(f"Invalid bucket_mode: {bucket_mode!r}")

        out["why_rank"] = (
            "Ranked by score = "
            + out["suitability"].map(lambda v: f"{v:.2f}")
            + " - ("
            + f"{alpha:.2f}"
            + " * "
            + out["competitiveness_index"].map(lambda v: f"{v:.2f}")
            + ") = "
            + out["score"].map(lambda v: f"{v:.2f}")
        )

        sal_mean = pd.to_numeric(out["sal_mean"], errors="coerce")
        pred_sal = pd.to_numeric(out["pred_sal"], errors="coerce")
        gap = sal_mean - pred_sal

        out["salary_context"] = (
            "Market mean = "
            + sal_mean.map(lambda v: f"{v:.2f}")
            + "; predicted for you = "
            + pred_sal.map(lambda v: f"{v:.2f}")
            + "; gap (market - predicted) = "
            + gap.map(lambda v: f"{v:.2f}")
        )

        return out

    top_best = _add_bucket_rank_salary_explanations(top_best, bucket_mode="best_now")
    top_stretch = _add_bucket_rank_salary_explanations(
        top_stretch, bucket_mode="stretch"
    )
    if include_scored_universe:
        scored_universe = _add_bucket_rank_salary_explanations(
            scored_universe, bucket_mode="mixed"
        )

    # -----------------------------
    # Add skill metrics (from candidate_jobs)
    # -----------------------------
    _require_cols(
        candidate_jobs,
        ["job_id", "skill_match_norm", "expected_missing_norm"],
        where="candidate_jobs",
    )
    gap_addon = candidate_jobs[
        ["job_id", "skill_match_norm", "expected_missing_norm"]
    ].copy()

    # Top-N are one-to-one; scored_universe is one-to-one as well (job_id unique)
    top_best = top_best.merge(gap_addon, how="left", on="job_id", validate="one_to_one")
    top_stretch = top_stretch.merge(
        gap_addon, how="left", on="job_id", validate="one_to_one"
    )
    if include_scored_universe:
        scored_universe = scored_universe.merge(
            gap_addon, how="left", on="job_id", validate="one_to_one"
        )

    # -----------------------------
    # Family-level missing / covered lists (via skill_mat)
    # -----------------------------
    user_skill_vector: pd.DataFrame = profile["derived"]["skill_vector"]
    if (
        not isinstance(user_skill_vector, pd.DataFrame)
        or user_skill_vector.shape[0] != 1
    ):
        raise ValueError(
            "profile['derived']['skill_vector'] must be a 1-row DataFrame of 0/1 family flags."
        )

    families = list(user_skill_vector.columns)

    _require_cols(skill_mat, ["job_id"], where="skill_prob_matrix")
    prob_cols = [f"{fam}_prob" for fam in families]
    missing_prob_cols = [c for c in prob_cols if c not in skill_mat.columns]
    if missing_prob_cols:
        raise KeyError(
            f"skill_prob_matrix missing required columns: {missing_prob_cols}"
        )

    req_matrix = skill_mat[["job_id"] + prob_cols].copy()
    req_matrix[prob_cols] = req_matrix[prob_cols] >= float(tau)

    rename_map = {f"{fam}_prob": fam for fam in families}
    req_matrix = req_matrix.rename(columns=rename_map)

    if req_matrix["job_id"].isna().any():
        raise ValueError("skill_prob_matrix contains null job_id values.")
    if not req_matrix["job_id"].is_unique:
        dup_n = int(req_matrix["job_id"].duplicated().sum())
        raise ValueError(
            f"skill_prob_matrix job_id is not unique ({dup_n} duplicates)."
        )

    u = user_skill_vector.iloc[0].astype(bool)
    R_all = req_matrix.set_index("job_id")[families].astype(bool)

    def _attach_family_lists(df: pd.DataFrame, *, where: str) -> pd.DataFrame:
        ids = df["job_id"].tolist()
        R = R_all.reindex(ids)

        if R.isna().any().any():
            missing_ids = R.index[R.isna().any(axis=1)].tolist()
            raise ValueError(
                f"{where}: skill_prob_matrix missing rows for job_id(s): {missing_ids[:10]}"
            )

        missing_mask = R & (~u)
        covered_mask = R & (u)

        missing_fams = missing_mask.apply(lambda r: r.index[r].tolist(), axis=1)
        covered_fams = covered_mask.apply(lambda r: r.index[r].tolist(), axis=1)

        out_df = df.copy()
        out_df["missing_families"] = missing_fams.values
        out_df["covered_families"] = covered_fams.values
        out_df["n_missing_families"] = out_df["missing_families"].map(len).astype(int)
        out_df["n_covered_families"] = out_df["covered_families"].map(len).astype(int)
        return out_df

    top_best = _attach_family_lists(top_best, where="top_best")
    top_stretch = _attach_family_lists(top_stretch, where="top_stretch")
    if include_scored_universe:
        scored_universe = _attach_family_lists(scored_universe, where="scored_universe")

    # -----------------------------
    # Light eval (fast invariants)
    # -----------------------------
    if validate:
        # Salary numeric sanity for top tables
        sal_mean_best = pd.to_numeric(top_best["sal_mean"], errors="coerce")
        pred_sal_best = pd.to_numeric(top_best["pred_sal"], errors="coerce")
        sal_mean_stretch = pd.to_numeric(top_stretch["sal_mean"], errors="coerce")
        pred_sal_stretch = pd.to_numeric(top_stretch["pred_sal"], errors="coerce")

        if sal_mean_best.isna().any() or pred_sal_best.isna().any():
            raise ValueError(
                "Non-numeric sal_mean/pred_sal detected in top_best after coercion."
            )
        if sal_mean_stretch.isna().any() or pred_sal_stretch.isna().any():
            raise ValueError(
                "Non-numeric sal_mean/pred_sal detected in top_stretch after coercion."
            )

        # Explanation columns should have no NaNs (top tables)
        for col in [
            "why_bucket",
            "why_rank",
            "salary_context",
            "missing_families",
            "covered_families",
        ]:
            if top_best[col].isna().any():
                raise ValueError(
                    f"NaNs detected in top_best explanation column '{col}'."
                )
            if top_stretch[col].isna().any():
                raise ValueError(
                    f"NaNs detected in top_stretch explanation column '{col}'."
                )

        # Metrics from candidate_jobs should be present and finite (top tables)
        for col in ["skill_match_norm", "expected_missing_norm"]:
            if top_best[col].isna().any() or (
                not np.isfinite(top_best[col].to_numpy()).all()
            ):
                raise ValueError(
                    f"Invalid values in top_best '{col}' after merge from candidate_jobs."
                )
            if top_stretch[col].isna().any() or (
                not np.isfinite(top_stretch[col].to_numpy()).all()
            ):
                raise ValueError(
                    f"Invalid values in top_stretch '{col}' after merge from candidate_jobs."
                )

        # Counts match list lengths (top tables)
        if (
            top_best["n_missing_families"] != top_best["missing_families"].map(len)
        ).any():
            raise ValueError("n_missing_families mismatch in top_best.")
        if (
            top_stretch["n_missing_families"]
            != top_stretch["missing_families"].map(len)
        ).any():
            raise ValueError("n_missing_families mismatch in top_stretch.")

        # Optional: validate scored_universe key fields (don’t overdo; it can be large)
        if include_scored_universe:
            _require_cols(
                scored_universe,
                [
                    "job_id",
                    "bucket",
                    "missing_families",
                    "covered_families",
                    "n_missing_families",
                    "n_covered_families",
                ],
                where="scored_universe_explained",
            )
            if scored_universe["job_id"].isna().any():
                raise ValueError(
                    "scored_universe_explained contains null job_id values."
                )
            if not scored_universe["job_id"].is_unique:
                dup_n = int(scored_universe["job_id"].duplicated().sum())
                raise ValueError(
                    f"scored_universe_explained job_id is not unique ({dup_n} duplicates)."
                )

    metric_glossary = {
        "job_id": "Unique identifier for the job posting/role row used throughout the project. Primary key for joins.",
        "title_rich": "Human-readable job title (normalized/cleaned form used for display).",
        "state": "Job location state (used for geo filtering and salary model features).",
        "Sector": "High-level sector label (business taxonomy; used in analysis/filters).",
        "Industry": "More specific industry label (nested under Sector; used in analysis/filters).",
        "Size": "Company size band (categorical; used in salary model features and analysis).",
        "sal_mean": "Market salary estimate for the role (posting-derived salary parsing; job-side 'expected' salary).",
        "pred_sal": "User-conditioned salary prediction for this job from the Chapter 1 salary model, using job codes + user skill PCs broadcast across candidate jobs.",
        "bucket": "Recommendation bucket from Chapter 4 logic. 'best_now' if competitiveness_index <= c_max; 'stretch' otherwise.",
        "competitiveness_index": "Barrier-to-entry composite score (higher = harder). Built in Chapter 3 using missingness/difficulty signals (and related components); used for bucketing and as a ranking penalty.",
        "suitability": "Fit score (higher = better). Built in Chapter 3 from user–job alignment (skills/embeddings/weights as configured). Primary positive term in ranking.",
        "score": "Final ranking score used within buckets: score = suitability - alpha * competitiveness_index. Higher ranks earlier.",
        "skill_match_norm": "Normalized skill-family match metric from Chapter 3 (higher = stronger alignment between the user's skill-family vector and the job's inferred requirements).",
        "expected_missing_norm": "Normalized expected-missingness metric from Chapter 3 (higher = larger/harder gaps vs job requirements; often incorporates rarity/difficulty weighting before normalization).",
        "why_bucket": "Human-readable rationale for why the job is in its bucket, based on competitiveness_index relative to c_max.",
        "why_rank": "Human-readable rationale for ranking, showing the score decomposition using suitability, competitiveness_index, and alpha.",
        "salary_context": "Human-readable salary comparison summarizing market mean (sal_mean), predicted salary for the user (pred_sal), and the gap (market - predicted).",
        "missing_families": "List of skill families inferred required for the job (probability >= tau) that are absent in the user's skill-family vector.",
        "covered_families": "List of skill families inferred required for the job (probability >= tau) that are present in the user's skill-family vector.",
        "n_missing_families": "Count of missing_families (quick scan without expanding list cells).",
        "n_covered_families": "Count of covered_families (quick scan without expanding list cells).",
        "explain_tau": "Threshold used to convert skill requirement probabilities into required/not-required for missing/covered family lists.",
        "alpha": "Ranking penalty weight applied to competitiveness_index in the score formula (higher alpha penalizes barrier more).",
        "c_max": "Competitiveness threshold used to split buckets into 'best_now' vs 'stretch'.",
    }

    tables_out: dict[str, Any] = {
        "top_best_explained": top_best,
        "top_stretch_explained": top_stretch,
    }
    if include_scored_universe:
        tables_out["scored_universe_explained"] = scored_universe

    return {
        "tables": tables_out,
        "metric_glossary": metric_glossary,
        "meta": {
            "tau": float(tau),
            "include_scored_universe": bool(include_scored_universe),
        },
    }
