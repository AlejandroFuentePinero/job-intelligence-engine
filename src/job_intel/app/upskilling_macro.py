# src/job_intel/app/upskilling_macro.py

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# -----------------------------
# Session access
# -----------------------------
def _get_upskilling_out_from_session() -> dict[str, Any] | None:
    """
    Expected storage pattern:
      - recommender page stores AppResult in st.session_state["result"]
      - AppResult.payload may store upskill output dict under a known key
    """
    res = st.session_state.get("result", None)
    if res is not None and hasattr(res, "payload") and isinstance(res.payload, dict):
        for k in ["upskill_out", "upskilling_out", "upskill", "upskilling"]:
            v = res.payload.get(k)
            if isinstance(v, dict):
                return v

    for k in ["upskill_out", "upskilling_out", "upskill"]:
        v = st.session_state.get(k)
        if isinstance(v, dict):
            return v

    pipe = st.session_state.get("pipeline_out", None)
    if isinstance(pipe, (tuple, list)) and len(pipe) >= 3 and isinstance(pipe[2], dict):
        return pipe[2]

    return None


def _scenario_to_family(s: str) -> str:
    s = str(s)
    if s == "baseline":
        return "baseline"
    if s.startswith("upskill_"):
        return s.replace("upskill_", "", 1)
    return s


# -----------------------------
# Style + name cleaning
# -----------------------------
import numpy as np  # add (you already use it in plotting)

_ACCENT = "#2F3E46"
_NEUTRAL = "#374151"
_TEXT = "#0B1220"
_GRID = "#E5E7EB"


def _set_mpl_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.edgecolor": _GRID,
            "axes.linewidth": 0.8,
            "grid.color": _GRID,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "axes.grid": False,
        }
    )


def _clean_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_GRID)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(colors=_NEUTRAL)
    ax.xaxis.label.set_color(_NEUTRAL)
    ax.yaxis.label.set_color(_NEUTRAL)
    ax.title.set_color(_TEXT)
    ax.grid(True, axis=grid_axis, alpha=0.9)


def _clean_skill_name(s: str) -> str:
    s = str(s)
    s = s.replace("__", " — ")
    s = s.replace("_prob", "")
    s = s.replace("_", " ")
    return s.strip()


# -----------------------------
# Tables (Upskilling)
# -----------------------------
def _build_missing_families_table(upskill_out: dict[str, Any]) -> pd.DataFrame:
    upskill_summary = upskill_out.get("upskill_summary")
    missing_dict = upskill_out.get("missing_dict", {}) or {}

    if not isinstance(upskill_summary, pd.DataFrame):
        return pd.DataFrame()

    summ = upskill_summary.reset_index().rename(
        columns={"upskill_scenario": "scenario"}
    )
    summ["skill family"] = summ["scenario"].map(_scenario_to_family)

    missing_fams = sorted([str(k) for k in missing_dict.keys()])
    summ = summ[summ["skill family"].isin(missing_fams)].copy()

    wanted = [
        "skill family",
        "upskill_impact_score",
        "promotion_rate",
        "demotion_rate",
        "mean_delta_score_stretch",
        "mean_delta_score_best",
        "p10_delta_score_best",
    ]
    out = summ[[c for c in wanted if c in summ.columns]].copy()

    if "upskill_impact_score" in out.columns:
        out = out.sort_values("upskill_impact_score", ascending=False)

    return out


def _build_top_recommendations_table(upskill_out: dict[str, Any]) -> pd.DataFrame:
    upskill_reco = upskill_out.get("upskill_recommendation")
    if not isinstance(upskill_reco, pd.DataFrame):
        return pd.DataFrame()

    df = upskill_reco.reset_index().rename(columns={"upskill_scenario": "scenario"})
    df["skill family"] = df["scenario"].map(_scenario_to_family)

    wanted = [
        "skill family",
        "upskill_impact_score",
        "promotion_rate",
        "demotion_rate",
        "mean_delta_score_stretch",
        "mean_delta_score_best",
        "p10_delta_score_best",
    ]
    out = df[[c for c in wanted if c in df.columns]].copy()

    if "upskill_impact_score" in out.columns:
        out = out.sort_values("upskill_impact_score", ascending=False)

    return out


def _build_token_table(
    recommendation_dict: dict[str, list[str]] | None,
) -> pd.DataFrame:
    if not isinstance(recommendation_dict, dict) or not recommendation_dict:
        return pd.DataFrame()

    rows = []
    for fam, toks in recommendation_dict.items():
        toks = toks or []
        rows.append(
            {
                "skill family": str(fam),
                "example skill tokens": ", ".join(map(str, toks)),
            }
        )
    return pd.DataFrame(rows).sort_values("skill family")


# -----------------------------
# Macro: skill similarity (co-learning)
# -----------------------------
def _repo_root() -> Path:
    # .../src/job_intel/app/upskilling_macro.py -> repo root = parents[3]
    return Path(__file__).resolve().parents[3]


def _default_similarity_edges_path() -> Path:
    return (
        _repo_root()
        / "data"
        / "processed"
        / "skill_similarity_matrix"
        / "skill_similarity_edges_k5_embeddings.csv"
    )


def _strip_prob_suffix(s: str) -> str:
    s = str(s)
    return s[:-5] if s.endswith("_prob") else s


def _load_similarity_edges() -> pd.DataFrame:
    """
    Loads undirected edge list with columns: skill_1, skill_2, similarity
    Cleans skill names to match Landscape plot naming.
    """
    path = _default_similarity_edges_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Skill similarity edge file not found at: {path}\n"
            "Expected: data/processed/skill_similarity_matrix/skill_similarity_edges_k5_embeddings.csv"
        )

    edges = pd.read_csv(path)
    need = {"skill_1", "skill_2", "similarity"}
    missing = need - set(edges.columns)
    if missing:
        raise KeyError(f"Similarity edges file missing columns: {sorted(missing)}")

    edges = edges.copy()
    edges["skill_1"] = edges["skill_1"].astype(str).apply(_clean_skill_name)
    edges["skill_2"] = edges["skill_2"].astype(str).apply(_clean_skill_name)
    edges["similarity"] = pd.to_numeric(edges["similarity"], errors="coerce")
    edges = edges.dropna(subset=["skill_1", "skill_2", "similarity"]).reset_index(
        drop=True
    )
    return edges


def _neighbors_for_focal(
    edges: pd.DataFrame, focal: str, *, top_k: int = 5
) -> pd.DataFrame:
    """
    edges: undirected edge list (skill_1, skill_2, similarity)
    focal: skill name (any raw format) -> cleaned to match edges
    """
    f = _clean_skill_name(focal)

    a = edges.loc[edges["skill_1"] == f, ["skill_2", "similarity"]].rename(
        columns={"skill_2": "co_learning_skill"}
    )
    b = edges.loc[edges["skill_2"] == f, ["skill_1", "similarity"]].rename(
        columns={"skill_1": "co_learning_skill"}
    )

    nei = pd.concat([a, b], ignore_index=True)
    nei = (
        nei.dropna(subset=["co_learning_skill"])
        .drop_duplicates(subset=["co_learning_skill"], keep="first")
        .sort_values("similarity", ascending=False)
        .head(int(top_k))
        .copy()
    )

    nei["focal_skill"] = f
    nei["rank"] = range(1, len(nei) + 1)
    return nei[["focal_skill", "co_learning_skill", "similarity", "rank"]]


def _build_neighbors_all(
    edges: pd.DataFrame, focals: list[str], *, top_k: int = 5
) -> pd.DataFrame:
    out = []
    for f in focals:
        out.append(_neighbors_for_focal(edges, focal=f, top_k=top_k))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# -----------------------------
# MODIFY _plot_grouped_colearning() to match Landscape style + safer label spacing
# -----------------------------
def _plot_grouped_colearning(
    nei_all: pd.DataFrame, *, focal_order: list[str], top_k: int = 5
) -> None:
    if nei_all is None or nei_all.empty:
        st.info("No co-learning neighbours found for the current recommended skills.")
        return

    # Clean and keep order
    focals = [_clean_skill_name(x) for x in focal_order]
    focals = [f for f in focals if f in set(nei_all["focal_skill"])]
    if not focals:
        st.info("No co-learning neighbours found for the current recommended skills.")
        return

    k = int(top_k)
    n_foc = len(focals)

    plot_df = (
        nei_all.sort_values(["focal_skill", "similarity"], ascending=[True, False])
        .groupby("focal_skill", as_index=False, group_keys=False)
        .head(k)
        .copy()
    )

    # Similarity scaling
    vmin = float(plot_df["similarity"].min())
    vmax = float(plot_df["similarity"].max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9

    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Minimal monochrome palette: map similarity to alpha in a single accent
    def _rgba(v: float):
        t = float(norm(v))
        return (*plt.matplotlib.colors.to_rgb(_ACCENT), 0.55 + 0.40 * t)

    base = list(range(n_foc))
    width = 0.14  # keep (your layout assumption)

    fig, ax = plt.subplots(figsize=(16, 8))

    max_h = float(plot_df["similarity"].max()) if not plot_df.empty else 1.0
    ax.set_ylim(0.0, max_h * 2.0)  # keep your headroom ratio

    pad = max_h * 0.03

    for j in range(k):
        rank = j + 1
        sub = plot_df[plot_df["rank"] == rank]

        y = []
        labels = []
        for f in focals:
            m = sub["focal_skill"] == f
            if m.any():
                y.append(float(sub.loc[m, "similarity"].iloc[0]))
                labels.append(
                    _clean_skill_name(sub.loc[m, "co_learning_skill"].iloc[0])
                )
            else:
                y.append(0.0)
                labels.append("")

        x = [b + (j - (k - 1) / 2) * width for b in base]
        colors = [_rgba(v) if v > 0 else (0, 0, 0, 0) for v in y]

        bars = ax.bar(
            x,
            y,
            width=width,
            color=colors,
            edgecolor=_GRID,
            linewidth=0.8,
        )

        for rect, lab, v in zip(bars, labels, y):
            if not lab or v <= 0:
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + pad,
                lab,
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=12,
                color=_TEXT,
                clip_on=True,
            )

    ax.set_title(
        "Macro co-learning: top skill neighbours for the recommended upskilling targets",
        loc="left",
        pad=14,
    )
    ax.set_xlabel("Recommended skill to upskill")
    ax.set_ylabel("Similarity")
    ax.set_xticks(base)
    ax.set_xticklabels(focals, color=_TEXT)

    _clean_axes(ax, grid_axis="y")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)  # keep extra top margin so labels don't collide

    st.pyplot(fig, clear_figure=True)


# -----------------------------
# Page
# -----------------------------
def render() -> None:
    st.title("Upskilling")
    with st.container(border=True):
        st.markdown(
            """
        This page converts your recommender results into an actionable upskilling plan. Start by running the **Recommender** first—Upskilling is generated from your current positioning and the roles you’re targeting. You can then explore the missing skill families for the most suitable jobs and see how each family contributes to improving your positioning. The engine will surface **three high-impact skills to learn now**, plus **co-learning suggestions** drawn from the macro market landscape to help you build complementary capability efficiently.
            """.strip()
        )

    _set_mpl_style()

    upskill_out = _get_upskilling_out_from_session()
    if upskill_out is None:
        st.info("Run the Recommender first (same session), then return here.")
        with st.expander("Debug: session_state keys"):
            st.json(list(st.session_state.keys()))
        return

    # --- Core outputs
    missing_table = _build_missing_families_table(upskill_out)
    top_table = _build_top_recommendations_table(upskill_out)
    token_df = _build_token_table(upskill_out.get("recommendation_dict"))

    # --- Explanation of weights + calculation
    meta = (
        upskill_out.get("meta", {}) if isinstance(upskill_out.get("meta"), dict) else {}
    )
    w = meta.get("weights", {}) if isinstance(meta.get("weights"), dict) else {}

    with st.expander("How the upskill_impact_score is computed", expanded=False):
        st.markdown(
            dedent(
                """
                The upskilling engine runs **counterfactual scenarios** on a **frozen job universe** (same job_ids).
                For each missing skill family, it injects representative tokens into your `skill_text`, reruns scoring,
                then compares each job’s **score** to baseline.

                **Key rates/metrics**
                - **promotion_rate**: fraction of baseline-stretch jobs that move into best_now after the upskill
                - **demotion_rate**: fraction of baseline-best_now jobs that fall into stretch after the upskill
                - **mean_delta_score_stretch**: average % score change among baseline-stretch jobs
                - **mean_delta_score_best**: average % score change among baseline-best_now jobs
                - **p10_delta_score_best**: 10th percentile % score change among baseline-best_now jobs (tail risk)

                **Composite ranking**
                - tail_penalty = abs(min(p10_delta_score_best, 0))
                - upskill_impact_score =
                  w_promote * promotion_rate
                  + w_stretch * mean_delta_score_stretch
                  + w_best * mean_delta_score_best
                  - w_demote * demotion_rate
                  - w_tail * tail_penalty
                """
            ).strip()
        )
        if w:
            st.markdown("**Weights (current run):**")
            st.json(w)

        guard = meta.get("demotion_tol", None)
        if guard is not None:
            st.markdown(f"**Guardrail:** demotion_rate ≤ {guard}")

    st.divider()

    st.subheader("All missing skill families (ranked)")
    if missing_table.empty:
        st.info(
            "No missing-family table found. (Check upskill_summary / missing_dict)."
        )
        with st.expander("Debug: upskill_out keys"):
            st.json(list(upskill_out.keys()))
    else:
        st.dataframe(missing_table, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Optimal upskilling prioritisation (what to learn next)")
    if top_table.empty:
        st.info("No top recommendations found.")
    else:
        st.dataframe(top_table, use_container_width=True, hide_index=True)

    with st.expander("Why these are the top recommendations", expanded=False):
        st.markdown(
            dedent(
                """
                These skill families are ranked to maximise **ROI**: they tend to (1) **promote** more roles from *stretch → best_now*,
                and/or (2) improve the average score of *stretch* roles, while keeping **downside** low.

                A small downside can occur because adding a skill family can shift the extracted skill profile in ways that
                slightly **reduces fit for some baseline best_now roles**—for example, making the profile look more specialised
                or moving it away from what those roles’ postings emphasise. That shows up as a non-zero **demotion_rate**
                or a negative tail effect (**p10_delta_score_best < 0**). The composite score explicitly penalises those risks,
                so the recommended families are the best “net gain with minimal harm” options under your current constraints.
                """
            ).strip()
        )

    st.subheader("Example skill tokens (for recommended families)")
    if token_df.empty:
        st.info("No recommendation_dict found.")
    else:
        st.dataframe(token_df, use_container_width=True, hide_index=True)

    # -----------------------------
    # Macro: co-learning suggestions
    # -----------------------------
    st.divider()
    st.subheader("Macro: Co-learning strength (skill similarity)")

    st.markdown(
        dedent(
            """
            These suggestions use a **skill–skill similarity graph** derived from **Node2Vec graph embeddings**
            learned over the labour-market skill ecosystem (built from the skill probability matrix).
            Skills that are close in embedding space tend to occur in similar contexts across the job market,
            so they are often efficient to learn together and can lift **average market suitability** more quickly.

            Higher means these skills tend to show up together across roles; not causal.
            """
        ).strip()
    )

    with st.expander("Interpretation", expanded=False):
        st.markdown(
            dedent(
                """
                The skill–skill network derived from Node2Vec embeddings reveals a coherent ecosystem structure in which skills cluster
                primarily by functional proximity and proficiency progression. The strongest connections link adjacent skill levels within
                the same domain (e.g., basic → intermediate ML, BI visualisation, database storage) and closely related transversal
                competencies (e.g., core soft skills and leadership). This indicates that the embedding space captures higher-order
                contextual similarity rather than simple co-occurrence, supporting its use as a relational representation of the
                labour-market skill landscape.
                """
            ).strip()
        )

    # Use the top-3 recommended families as focals
    focals: list[str] = []
    if not top_table.empty and "skill family" in top_table.columns:
        focals = top_table["skill family"].astype(str).head(3).tolist()

    focals = [_clean_skill_name(f) for f in focals]
    focals = [f for f in focals if f and f != "baseline"]
    if not focals:
        st.info(
            "No recommended skill families available to generate macro co-learning neighbours."
        )
        return

    try:
        edges = _load_similarity_edges()
        nei_all = _build_neighbors_all(edges, focals, top_k=5)
        _plot_grouped_colearning(nei_all, focal_order=focals, top_k=5)

        with st.expander("Show neighbour table", expanded=False):
            st.dataframe(
                nei_all.sort_values(
                    ["focal_skill", "similarity"], ascending=[True, False]
                ),
                use_container_width=True,
                hide_index=True,
            )

    except Exception as e:
        st.error(f"Failed to build macro co-learning suggestions: {e}")
