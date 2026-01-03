from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.job_intel.config import (
    PROCESSED_DATA_DIR,
    CH1_PROCESSED_SALARY_MODEL_PCA_DF,
)


def _assets_dir() -> Path:
    return PROCESSED_DATA_DIR / "ch5_assets"


# -----------------------------
# Matplotlib styling (minimal + professional)
# -----------------------------
_ACCENT = "#2F3E46"  # deep blue-grey
_NEUTRAL = "#6B7280"  # muted grey
_GRID = "#E5E7EB"  # light grey
_TEXT = "#111827"  # near-black


def _set_mpl_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": _GRID,
            "axes.linewidth": 0.8,
            "grid.color": _GRID,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "axes.grid": False,  # opt-in per plot
        }
    )


def _clean_axes(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_GRID)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(colors=_NEUTRAL)
    ax.xaxis.label.set_color(_NEUTRAL)
    ax.yaxis.label.set_color(_NEUTRAL)
    ax.title.set_color(_TEXT)
    ax.grid(True, axis=grid_axis, alpha=0.9)


# -----------------------------
# UI helpers
# -----------------------------
def _top_box(body_md: str) -> None:
    """
    Keep a visible (non-collapsible) box directly under the title.
    """
    with st.container(border=True):
        st.markdown(body_md)


def _interp_expander(title: str, body_md: str, *, expanded: bool = False) -> None:
    """
    Collapsible interpretation / details block.
    """
    with st.expander(title, expanded=expanded):
        st.markdown(body_md)


# -----------------------------
# Skill name cleaning (shared)
# -----------------------------
def _clean_skill_name(s: str) -> str:
    s = str(s)
    s = s.replace("__", " — ")
    s = s.replace("_prob", "")
    s = s.replace("_", " ")
    return s.strip()


# -----------------------------
# Loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def _load_fairness_tables() -> tuple[pd.DataFrame, dict]:
    assets = _assets_dir()
    group = pd.read_csv(assets / "fairness_group_summary_long.csv")
    box = json.loads((assets / "fairness_residual_box_stats.json").read_text())
    return group, box


@st.cache_data(show_spinner=False)
def _load_residuals_series() -> pd.Series:
    df = pd.read_csv(
        PROCESSED_DATA_DIR / "df_with_residuals.csv", usecols=["residuals"]
    )
    return pd.to_numeric(df["residuals"], errors="coerce").dropna()


@st.cache_data(show_spinner=False)
def _load_skill_value_index() -> pd.DataFrame:
    p = _assets_dir() / "skill_value_index.csv"
    df = pd.read_csv(p)

    if not {"skill", "value"}.issubset(df.columns):
        raise ValueError("skill_value_index.csv must have columns: skill, value")

    # Clean skill names to match the similarity plot
    df["skill"] = df["skill"].astype(str).map(_clean_skill_name)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["skill", "value"]).copy()

    # If cleaning creates duplicates, collapse deterministically
    df = (
        df.groupby("skill", as_index=False)
        .agg(value=("value", "mean"))
        .sort_values("value", ascending=False)
        .reset_index(drop=True)
    )
    return df


@st.cache_data(show_spinner=False)
def _load_ch1_lookup_tables() -> dict[str, pd.DataFrame]:
    """
    Load lookup tables mapping *_code -> human label columns.
    """
    df_ch0 = pd.read_csv(CH1_PROCESSED_SALARY_MODEL_PCA_DF)

    def _dedup(code_col: str, label_col: str) -> pd.DataFrame:
        out = (
            df_ch0[[code_col, label_col]]
            .drop_duplicates(subset=code_col)
            .dropna(subset=[code_col])
            .copy()
        )
        out[code_col] = pd.to_numeric(out[code_col], errors="coerce")
        out = out.dropna(subset=[code_col]).sort_values(code_col)
        out[label_col] = out[label_col].astype(str)
        return out

    return {
        "size": _dedup("size_code", "Size"),
        "sector": _dedup("sector_code", "Sector"),
        "state": _dedup("state_code", "state"),
        "ownership": _dedup("ownership_code", "ownership_clean"),
        "seniority": _dedup("seniority_code", "seniority_combined"),
        "title": _dedup("title_rich_code", "title_rich"),
    }


@st.cache_data(show_spinner=False)
def _load_shap_explanation() -> Any:
    """
    Loads shap_values artefact saved to assets dir.

    Expected filename:
      - shap_salary_explanation.npz

    Required keys:
      - values, feature_names
    Optional:
      - data, base_values
    """
    import shap  # type: ignore

    assets = _assets_dir()
    p = assets / "shap_salary_explanation.npz"
    if not p.exists():
        raise FileNotFoundError(f"Could not find {p.name} in {assets}")

    obj = np.load(p, allow_pickle=True)

    if "values" not in obj or "feature_names" not in obj:
        raise ValueError(
            "shap_salary_explanation.npz must contain at least: values, feature_names"
        )

    values = np.asarray(obj["values"], dtype=float)
    feature_names = [str(x) for x in obj["feature_names"].tolist()]

    data = None
    if "data" in obj:
        data = np.asarray(obj["data"])

    base_values = None
    if "base_values" in obj:
        base_values = obj["base_values"]

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
    )


@st.cache_data(show_spinner=False)
def _load_skill_similarity_matrix() -> pd.DataFrame:
    p = _assets_dir() / "skill_similarity_matrix.csv"
    df = pd.read_csv(p)

    if "skill" not in df.columns:
        raise ValueError("skill_similarity_matrix.csv must include a 'skill' column")

    df = df.set_index("skill")
    df.index = df.index.map(_clean_skill_name)
    df.columns = [_clean_skill_name(c) for c in df.columns]
    df.index = df.index.astype(str)
    df.columns = [str(c) for c in df.columns]

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    if set(df.columns) == set(df.index):
        df = df.loc[df.index, df.index]

    arr = df.to_numpy(copy=True)
    np.fill_diagonal(arr, np.nan)
    df = pd.DataFrame(arr, index=df.index, columns=df.columns)
    return df


# -----------------------------
# Plot helpers
# -----------------------------
def _plot_residual_hist(residuals: pd.Series, bins: int = 40) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.hist(
        residuals,
        bins=bins,
        color=_ACCENT,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
    )
    ax.axvline(0, linewidth=1, color=_GRID)
    ax.set_title("Residual distribution", loc="left", pad=10)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    _clean_axes(ax, grid_axis="x")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _plot_group_bars(df: pd.DataFrame, metric: str, title: str) -> None:
    d = df.copy()
    d["group_value"] = d["group_value"].astype(str)

    n = len(d)
    fig_h = min(18.0, max(5.5, 0.28 * n + 1.5))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    ax.barh(d["group_value"][::-1], d[metric][::-1], color=_ACCENT, alpha=0.92)
    ax.axvline(0, linewidth=1, color=_GRID)

    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel(metric)
    ax.set_ylabel("")

    _clean_axes(ax, grid_axis="x")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _plot_skill_value_bars(df: pd.DataFrame) -> None:
    d = df.sort_values("value", ascending=False).copy()

    n = len(d)
    fig_h = min(14.0, max(5.5, 0.32 * n + 1.5))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    ax.barh(d["skill"].astype(str)[::-1], d["value"][::-1], color=_ACCENT, alpha=0.92)
    ax.axvline(0, linewidth=1, color=_GRID)

    ax.set_title("Global Skill Value Index (GSVI)", loc="left", pad=10)
    ax.set_xlabel("Value (higher = more associated with higher salary)")
    ax.set_ylabel("")

    _clean_axes(ax, grid_axis="x")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _global_shap_bar_fig(shap_values: Any) -> plt.Figure:
    values = np.asarray(shap_values.values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    names = list(shap_values.feature_names)

    mean_abs = np.mean(np.abs(values), axis=0)
    order = np.argsort(mean_abs)[::-1]

    ordered_names = [names[i] for i in order]
    ordered_vals = mean_abs[order]

    n = len(ordered_names)
    fig_h = min(13.5, max(7.9, 0.28 * n + 1.5))

    fig, ax = plt.subplots(figsize=(7.8, fig_h))
    ax.barh(ordered_names[::-1], ordered_vals[::-1], color=_ACCENT, alpha=0.92)
    ax.set_title(
        "Global SHAP — mean(|contribution|) by feature",
        loc="left",
        pad=10,
        fontsize=20,
    )
    ax.set_xlabel("Mean absolute SHAP value")

    _clean_axes(ax, grid_axis="x")

    _apply_fig_text_sizes(fig, title_fs=16, label_fs=13, tick_fs=12)

    fig.tight_layout()
    return fig


def _beeswarm_fig(shap_values: Any, max_display: int = 20) -> plt.Figure:
    """
    Try notebook beeswarm call; if it fails due to constant features, drop constants for beeswarm only.
    """
    import shap  # type: ignore

    title = "Beeswarm (feature impact distribution)"

    try:
        plt.figure(figsize=(7.8, 5.8))
        shap.plots.beeswarm(shap_values, max_display=int(max_display), show=False)

        ax = plt.gca()
        ax.set_title(title, loc="center", pad=10)

        fig = plt.gcf()
        _apply_fig_text_sizes(fig, title_fs=16, label_fs=13, tick_fs=12)
        fig.tight_layout()
        return fig
    except Exception:
        pass

    try:
        data = shap_values.data
        values = np.asarray(shap_values.values, dtype=float)
        names = list(shap_values.feature_names)

        if data is None:
            plt.figure(figsize=(7.8, 5.8))
            shap.plots.beeswarm(shap_values, max_display=int(max_display), show=False)

            ax = plt.gca()
            ax.set_title(title, loc="center", pad=10)

            fig = plt.gcf()
            _apply_fig_text_sizes(fig, title_fs=16, label_fs=13, tick_fs=12)
            fig.tight_layout()
            return fig

        data_np = np.asarray(data)
        with np.errstate(all="ignore"):
            v = np.nanmax(data_np, axis=0) - np.nanmin(data_np, axis=0)
        keep = np.where(np.isfinite(v) & (v > 0))[0]

        if keep.size == 0:
            fig, ax = plt.subplots(figsize=(7.8, 5.8))
            ax.set_title(title, loc="center", pad=10)
            ax.text(
                0.5,
                0.5,
                "Beeswarm unavailable (all displayed features are constant).",
                ha="center",
                va="center",
                color=_NEUTRAL,
            )
            ax.axis("off")
            return fig

        expl = shap.Explanation(
            values=values[:, keep],
            base_values=shap_values.base_values,
            data=data_np[:, keep],
            feature_names=[names[i] for i in keep],
        )

        plt.figure(figsize=(7.8, 5.8))
        shap.plots.beeswarm(expl, max_display=int(max_display), show=False)

        ax = plt.gca()
        ax.set_title(title, loc="center", pad=10)

        fig = plt.gcf()
        _apply_fig_text_sizes(fig, title_fs=16, label_fs=13, tick_fs=12)
        fig.tight_layout()
        return fig

    except Exception as e:
        fig, ax = plt.subplots(figsize=(7.8, 5.8))
        ax.set_title(title, loc="center", pad=10)
        ax.text(
            0.5,
            0.5,
            f"Could not render beeswarm: {e}",
            ha="center",
            va="center",
            color=_NEUTRAL,
        )
        ax.axis("off")
        return fig


def _apply_fig_text_sizes(
    fig: plt.Figure,
    *,
    title_fs: int = 16,
    label_fs: int = 13,
    tick_fs: int = 12,
) -> None:
    # Apply to all axes (main + colorbar axis in SHAP)
    for ax in fig.axes:
        # Titles/labels (safe even if empty)
        ax.title.set_fontsize(title_fs)
        ax.xaxis.label.set_size(label_fs)
        ax.yaxis.label.set_size(label_fs)

        # Tick labels
        ax.tick_params(axis="both", labelsize=tick_fs)


def _local_shap_frames(
    shap_values: Any,
    *,
    feature: str,
    code_col: str,
    label_col: str,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    names = list(shap_values.feature_names)
    if feature not in names:
        raise KeyError(f"{feature} not found in shap feature_names")

    idx = {n: i for i, n in enumerate(names)}[feature]

    codes = shap_values.data[:, idx] if shap_values.data is not None else None
    if codes is None:
        raise ValueError(
            "shap_values.data is missing; cannot build local categorical plots."
        )

    codes = pd.to_numeric(pd.Series(codes), errors="coerce")
    mask = codes.notna()
    codes = codes.loc[mask].astype(float).astype(int)

    shap_col = pd.to_numeric(
        pd.Series(np.asarray(shap_values.values, dtype=float)[:, idx]), errors="coerce"
    )
    shap_col = shap_col.loc[mask]

    df = pd.DataFrame({code_col: codes.values, "shap_value": shap_col.values})
    df = df.merge(lookup[[code_col, label_col]], on=code_col, how="left")
    df[label_col] = df[label_col].fillna(df[code_col].astype(str)).astype(str)
    return df


def _plot_local_bar_and_box(
    df: pd.DataFrame,
    *,
    label_col: str,
    title: str,
    top_n: int = 15,
) -> plt.Figure:
    summ = (
        df.groupby(label_col, as_index=False)
        .agg(mean_shap=("shap_value", "mean"), n=("shap_value", "size"))
        .sort_values("mean_shap", ascending=False)
    )

    summ["abs_mean"] = summ["mean_shap"].abs()
    summ = summ.sort_values("abs_mean", ascending=False).head(int(top_n))
    summ = summ.sort_values("mean_shap", ascending=False)

    order = summ[label_col].tolist()
    data = [
        df.loc[df[label_col] == lab, "shap_value"].astype(float).values for lab in order
    ]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.5, 5.2))

    # Mean bar
    axes[0].barh(order[::-1], summ["mean_shap"].values[::-1], color=_ACCENT, alpha=0.92)
    axes[0].axvline(0, linewidth=1, color=_GRID)
    axes[0].set_title(f"{title} — mean SHAP (premium / penalty)", loc="left", pad=10)
    axes[0].set_xlabel("Mean SHAP value")
    axes[0].set_ylabel("")
    _clean_axes(axes[0], grid_axis="x")

    _STROKE = "#989DA3"

    # Box (distribution)
    bp = axes[1].boxplot(
        data[::-1],
        labels=order[::-1],
        vert=False,
        showfliers=False,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": _TEXT, "linewidth": 1.2},
        boxprops={"edgecolor": _STROKE, "linewidth": 1.1},
        whiskerprops={"color": _STROKE, "linewidth": 1.1},
        capprops={"color": _STROKE, "linewidth": 1.1},
    )
    for box in bp["boxes"]:
        box.set_facecolor("#C3AB34")  # very light fill

    axes[1].axvline(0, linewidth=1, color=_GRID)
    axes[1].set_title(f"{title} — SHAP distribution", loc="left", pad=10)
    axes[1].set_xlabel("SHAP value")
    axes[1].set_ylabel("")
    _clean_axes(axes[1], grid_axis="x")

    fig.tight_layout()
    return fig


def _order_by_similarity(sim: pd.DataFrame) -> list[str]:
    """
    Returns an ordering of skills that groups similar skills together
    using hierarchical clustering on distance = 1 - similarity.

    If scipy is unavailable, falls back to current order.
    """
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
    except Exception:
        return sim.index.tolist()

    m = sim.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(m, 1.0)

    d = 1.0 - m
    d = np.clip(d, 0.0, 2.0)
    np.fill_diagonal(d, 0.0)

    condensed = squareform(d, checks=False)
    Z = linkage(condensed, method="average", optimal_ordering=True)
    order_idx = leaves_list(Z)

    labels = sim.index.to_numpy()
    return labels[order_idx].tolist()


def _orient_order_top_left(
    sim: pd.DataFrame, order: list[str], k: int = 8
) -> list[str]:
    """
    Choose between order and reversed(order) to push high-similarity structure
    towards the top-left (heuristic: maximize mean similarity in the top-left k×k block).
    """
    k = max(3, min(int(k), len(order)))
    A = sim.loc[order, order].to_numpy(dtype=float, copy=True)
    B = sim.loc[list(reversed(order)), list(reversed(order))].to_numpy(
        dtype=float, copy=True
    )

    np.fill_diagonal(A, np.nan)
    np.fill_diagonal(B, np.nan)

    score_a = float(np.nanmean(A[:k, :k]))
    score_b = float(np.nanmean(B[:k, :k]))

    return order if score_a >= score_b else list(reversed(order))


def _plot_skill_similarity_heatmap(
    sim: pd.DataFrame, *, triangle: str = "lower"
) -> None:
    """
    Plots only one half of the symmetric similarity matrix and orders skills
    to group similar skills together.

    triangle: "lower" or "upper"
    """
    # Comment out to order skills by similarity
    # order = _order_by_similarity(sim)
    # order = _orient_order_top_left(sim, order, k=8)
    # sim = sim.loc[order, order]

    mat = sim.to_numpy(dtype=float, copy=True)

    np.fill_diagonal(mat, np.nan)
    if triangle == "lower":
        mat[np.triu_indices_from(mat, k=1)] = np.nan
    elif triangle == "upper":
        mat[np.tril_indices_from(mat, k=-1)] = np.nan
    else:
        raise ValueError("triangle must be 'lower' or 'upper'")

    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        st.info("Skill similarity matrix has no finite values to plot.")
        return

    vmin = float(np.nanpercentile(finite, 5))
    vmax = float(np.nanpercentile(finite, 95))
    if np.isclose(vmin, vmax):
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))

    n = sim.shape[0]
    fig_w = min(12.5, max(7.5, 0.22 * n + 4.0))
    fig_h = min(12.0, max(7.0, 0.22 * n + 3.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        mat,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_title("Skill similarity matrix", loc="left", pad=10)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(
        sim.columns.tolist(), rotation=90, ha="center", va="top", color=_NEUTRAL
    )
    ax.set_yticklabels(sim.index.tolist(), color=_NEUTRAL)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Cosine similarity", color=_NEUTRAL)
    cbar.ax.yaxis.set_tick_params(color=_NEUTRAL)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=_NEUTRAL)

    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# Interpretation text
# -----------------------------
_SHAP_INTRO = """
**SHAP** (SHapley Additive exPlanations) attributes a model’s prediction to individual features by computing each feature’s
marginal contribution across all possible feature subsets.

For each observation and feature, the SHAP value quantifies how much that feature pushes the prediction up or down relative
to the model’s baseline. The explanation is additive: **baseline + sum(SHAP values) = prediction**.
""".strip()

_SHAP_GLOBAL_INTERP = """
##### Global SHAP
The global SHAP bar plot indicates that **structural job attributes dominate salary predictions**.
The strongest drivers are **geographic location** (`state_code`) and **role richness** (`title_rich_code`).
**Industry sector** (`sector_code`) also contributes meaningfully, while **company size** (`size_code`) and
**ownership** (`ownership_code`) tend to have more moderate effects.

**Skills enter the model via PCA components** (`skill_PC*`) to reduce overlap and multicollinearity across many
correlated skill signals. Skills matter **collectively**, but their influence is typically **distributed across PCs**
and smaller than the dominant location/title effects.

**Skill PCs (interpretation guide)**

- **Skill_PC1:** foundational technical infrastructure (core programming, databases, pipelines, cloud)
- **Skill_PC2:** analytics + BI orientation (analysis/reporting; “analyst-profile” bundle)
- **Skill_PC3:** ML/AI + advanced analytics intensity
- **Skill_PC4:** leadership/workflow + intermediate infrastructure (coordination/management-adjacent bundle)
- **Skill_PC5:** weak / redundant mixed signal (little standalone meaning)
- **Skill_PC6:** basic analytics/BI + mid-level pipeline skills (niche configuration)
- **Skill_PC7:** general programming + basic analytics contrasted against ML-oriented depth
- **Skill_PC8:** hybrid bundle (basic analytics + soft skills + intermediate infrastructure)
- **Skill_PC9:** weak / near-zero signal (no clear interpretable bundle)
- **Skill_PC10:** productivity + workflow tooling (process/tooling bundle)

##### Beeswarm (what it adds)

The beeswarm plot shows how feature effects vary across jobs, and it makes clear that geographic location (state_code) and role richness (title_rich_code) dominate the model’s behaviour. These features have the widest, strongly bidirectional spreads, meaning different categories can push predicted salary substantially up or down and effectively define distinct salary regimes. Industry sector (sector_code), company size (size_code), and ownership (ownership_code) have more moderate, narrower distributions, acting as stabilising contextual shifts rather than extreme drivers. The skill PCA components show smaller magnitudes and tighter clustering, indicating skills tend to provide conditional, incremental uplift within the structural constraints set by geography and role. The near-zero effect of seniority_code suggests seniority is largely absorbed by richer title representations and correlated structural features, reinforcing the hierarchy that structure comes first and skills refine within it.
""".strip()

_LOCAL_INTERP = {
    "sector_code": (
        "Sector exhibits the largest absolute salary effects among categorical variables, indicating industry context is a major structural driver of pay. "
        "High-premium sectors typically reflect high revenue-per-employee and technical scarcity; strong penalties often reflect institutional wage constraints "
        "or lower market competition rather than individual skill deficits."
    ),
    "state_code": (
        "Geographic location is a dominant structural determinant of pay. Large premiums/penalties reflect cost-of-living, labour-market competition, "
        "regional industry concentration, and pay norms. This is not a skill effect: similar roles can be paid very differently by location."
    ),
    "title_rich_code": (
        "Job title is the most discriminative categorical variable, producing clean separation between premiums and penalties. "
        "The model learns a clear occupational hierarchy: ML/AI-heavy scientist/engineer titles at the top, analyst titles at the bottom, with data scientist/engineer in between."
    ),
}

_SKILL_MODELS_SNAPSHOT = {
    "models": "27 skill-family classifiers (LightGBM)",
    "metrics_used": ["ROC AUC", "PR AUC", "Brier"],
    "roc_auc_typical_range": "0.88–0.95 (min ~0.80, max 1.00)",
    "pr_auc_summary": "mean ~0.75, median ~0.77; many skills 0.85–0.95",
    "brier_typical_range": "0.06–0.12",
    "pattern_by_prevalence": {
        "high_prevalence_skills": "PR AUC ~0.90–1.00",
        "moderate_prevalence_skills": "PR AUC ~0.70–0.85",
        "low_prevalence_advanced_skills": "PR AUC ~0.35–0.55",
    },
}


# -----------------------------
# Page
# -----------------------------
def render() -> None:
    st.title("Landscape")
    _set_mpl_style()

    st.markdown(
        """
This Landscape page summarises the global market signal learned from the job-ad dataset: which job attributes most strongly shape salary, where persistent pay premiums/penalties remain after controls, and which skill bundles align with higher predicted pay. It combines global SHAP, residual (fairness) diagnostics, and the Global Skill Value Index to show that structural factors (role/title, location, sector, company context) dominate, with skills refining outcomes within those regimes. The goal is to provide the context that makes the Recommender and Upskilling outputs interpretable—explaining the “shape” of the market before positioning an individual user inside it.
        """.strip()
    )

    _interp_expander(
        "Learn more about the landsacape",
        """
        "The job market landscape is the project’s “map” of how data roles are priced and differentiated in the real world—what the market consistently rewards, what it discounts, and which constraints are structural rather than personal. It is the backbone of the recommender: the system is not guessing in a vacuum, it is positioning you inside this learned landscape so the recommendations are explainable rather than arbitrary. The strongest result in this project is that salary is driven first by structural context—role semantics (enriched job title), location (state), sector, and company context—and only then refined by skill bundles, meaning skills usually move you within a role/location regime more than they let you “escape” it. Residual (fairness) analysis reinforces this by showing persistent premiums and penalties by employer type and scale: large/public employers tend to pay above expectation, while small/private and especially non-profit roles underpay relative to comparable jobs. The model also learns an occupational hierarchy that will feel familiar: ML/AI-heavy scientist/engineer titles cluster in higher-paying regimes, while analyst-oriented titles cluster in lower-paying regimes, even after controlling for other factors. Skills appear as structured bundles (PCA components) with threshold-like effects—broad core infra/programming behaves like a gatekeeper, while ML/modelling depth is where the clearest uplift emerges. Use this page to understand the market context first, then interpret your personalised recommendations as “where you sit on the map” and your upskilling plan as the smallest set of moves that most reliably shifts your position toward better role regimes and better pay."

            """.strip(),
        expanded=True,
    )

    # if st.button("Reload assets"):
    #    st.cache_data.clear()
    #    st.rerun()
    st.divider()

    st.markdown("## Landscape view menu")
    view = st.selectbox(
        "Landscape view menu",
        [
            "Fairness insights",
            "Skill value ranking and similarity",
            "SHAP importance",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    # -----------------------------
    # Skill value ranking
    # -----------------------------
    if view == "Skill value ranking and similarity":
        st.subheader("Global skill value ranking")

        _interp_expander(
            "What this is / How to read it",
            """
**What this is:** The **Global Skill Value Index (GSVI)** is a model-implied ranking of which *skill families* tend to appear in
higher-paying roles **after controlling** for title, seniority, sector, location, and company attributes.

**How to read it:** Higher values mean the skill family is more associated with higher predicted salaries in this dataset.
It’s **global** (not personalised) and **not causal**—use the Recommender/Upskilling pages to translate this into feasibility and ROI.

**Interpretation Guidelines:**
- Rankings reflect the structure of the observed job market data
- Results depend on how skills co-occur in real job postings
- Skills should be interpreted as **signals within bundles**, not isolated levers
            """.strip(),
            expanded=False,
        )

        svi = _load_skill_value_index()
        _plot_skill_value_bars(svi)
        st.caption(
            "Skill value signal derived from the fitted salary response model. GSVI captures relative, conditional associations with predicted salary."
        )

        _interp_expander(
            "Interpretation (GSVI)",
            """
The **Global Skill Value Index (GSVI)** is a model-implied, **unitless** ranking of how each skill band is
associated with predicted salary **after controlling** for job title, seniority, sector/industry, location, and company attributes.

It is computed by **back-projecting** the salary model’s PCA component effects onto individual skills using the PCA loadings
(no new model fitting, no SHAP recomputation).

Higher values mean the skill band is more strongly associated with higher salaries in this dataset; negative values mean the opposite.
This is a **global descriptive market signal**, not causal and not monetary.
            """.strip(),
            expanded=False,
        )

        st.divider()
        st.subheader("Skill similarity heatmap")

        sim = _load_skill_similarity_matrix()
        _plot_skill_similarity_heatmap(sim, triangle="lower")

        _interp_expander(
            "Interpretation (Skill similarity matric)",
            """
This heatmap shows **pairwise skill similarity** computed as **cosine similarity** between **Node2Vec skill embeddings**
learned from the job–skill graph. Similarity reflects **shared labour-market context**: skills that appear in similar roles,
or represent adjacent proficiency progression within a domain, sit closer in embedding space.

High similarity indicates **functional relatedness** (often basic → intermediate → advanced within the same domain, or tightly coupled transversal skills),
not raw frequency and not a causal relationship. Use it to identify natural **skill bundles**, “co-learning neighbours”, and gateway skill structure.
            """.strip(),
            expanded=False,
        )

        with st.expander("Skill models performance snapshot", expanded=False):
            st.markdown(
                """
These numbers summarise the **quality of the underlying skill signals** that feed the Skill Value ranking and the recommender.
Lower PR AUC for rare, advanced skills is expected (class imbalance); Brier indicates probability calibration.
                """.strip()
            )
            st.json(_SKILL_MODELS_SNAPSHOT)

        st.caption(
            "Scope: global aggregation only. No causal claims, no uncertainty estimates."
        )
        return

    # -----------------------------
    # Fairness residuals
    # -----------------------------
    if view == "Fairness insights":
        st.subheader("Fairness (model residuals)")
        group, box = _load_fairness_tables()

        _interp_expander(
            "What this is / How to read it",
            """
**What this is:** *Residuals* are the difference between the observed salary and the model’s predicted salary.
They highlight where pay is systematically **above** (positive residual) or **below** (negative residual) what the model expects
after controlling for job characteristics.

**How to read it:** This is a **descriptive fairness lens**, not a causal claim. It helps surface structural pay gradients
(e.g., location/sector/title patterns) that persist even after accounting for skills and job attributes.
            """.strip(),
            expanded=False,
        )

        st.markdown("### Overall distribution")
        residuals = _load_residuals_series()
        _plot_residual_hist(residuals, bins=40)

        with st.expander("Box summary stats", expanded=False):
            st.json(box)

        with st.expander(
            "Salary model performance & training snapshot", expanded=False
        ):
            st.json(
                {
                    "model": "XGBRegresssor",
                    "cv_folds": 3,
                    "grid_candidates": 175,
                    "total_fits": 525,
                    "best_params": {
                        "learning_rate": 0.05,
                        "max_depth": 2,
                        "n_estimators": 100,
                    },
                    "r2_test": 0.30,
                    "r2_train": 0.35,
                    "rmse": 31541.32,
                    "mae": 24977.34,
                }
            )

        st.divider()
        st.markdown("### Group analysis")

        group_types = sorted(group["group_type"].dropna().unique().tolist())

        _INTERPRETATION = {
            "location": "Location residuals summarise model-adjusted over/underpayment by state. The unweighted residual means show how each state pays relative to model expectations, revealing that a few states (e.g., CA, NY) slightly overpay while many smaller states underpay. However, this view is heavily influenced by states with very small sample sizes, which can exaggerate or distort the visual pattern. The size-weighted residual plot corrects this by scaling each state’s contribution by its proportion of the total data, producing a more realistic picture of overall geographic fairness. In this weighted view, large states dominate the signal—as they should—and small states contribute almost no bias. Together, the two plots show both the raw per-state deviations and the true market-level impact of those deviations.",
            "sector": "Sector residuals summarise model-adjusted over/underpayment by industry context. The unweighted plot shows that sectors requiring continuous technical updating—such as Information Technology, Energy, and Biotech/Pharma—tend to pay above the model’s expectations, suggesting a premium for sectors where skills evolve rapidly and staying current carries high value. In contrast, more stable or mature sectors like Finance, Insurance, and Administrative Services tend to underpay relative to predicted norms, which may reflect lower technical intensity, slower skill turnover, or tighter structural salary bands. Sectors with inherently low margins or low barriers to entry—such as Education, Non-Profit, and Food Services—consistently fall below expectations as well. The weighted plot confirms that the sectors with genuine market impact are those with both high skill requirements and large job volumes, primarily IT and Biotech. Overall, the sector fairness analysis shows that technical dynamism and skill intensity appear to be key drivers of over- or under-payment once job characteristics are controlled for.",
            "company_size": "Company size residuals summarise model-adjusted over/underpayment by employer scale. The fairness analysis shows a strong structural gradient across company sizes: the largest firms (10,000+ employees) consistently pay far above model expectations, even after controlling for title, skills, sector, location, and seniority. Mid-sized organisations (1001–5000 and 51–200 employees) show mild positive deviations, indicating they remain competitive but less aggressive than mega-corporations. Smaller companies—especially those under 500 employees—systematically underpay relative to predicted norms, likely reflecting tighter budgets, lower revenue stability, and reduced salary-band flexibility. The “Unknown” category sits near zero, suggesting it is a mix of companies of varied sizes with no clear trend. Overall, company size fairness results follow well-established patterns: larger organisations can afford higher compensation, have stronger competition for talent, and offer structural salary premiums beyond what job characteristics alone would predict.",
            "ownership": "Ownership residuals summarise model-adjusted over/underpayment by public/private/nonprofit/government context. The ownership fairness results show that public companies pay well above model expectations, indicating strong compensation competitiveness after adjusting for job, skill, sector, and location differences. Government roles also display positive deviations, though at a lower magnitude, reflecting stable but modestly premium salary structures relative to similar private-sector roles. The private sector, despite dominating the dataset, systematically underpays relative to predicted norms, suggesting tighter salary bands and stronger cost-efficiency pressures. Nonprofit organisations emerge as the lowest-paying category once job characteristics are controlled for, consistent with their mission-driven, lower-budget operating models. Overall, ownership type is a clear structural driver of salary variation: public > government > private > nonprofit in terms of adjusted salary premiums.",
            "seniority": "Seniority residuals summarise model-adjusted over/underpayment by responsibility level. The seniority fairness analysis shows a strong salary gradient: principal and manager roles are paid well above model expectations, even after controlling for title, skills, sector, and company attributes. Senior roles also show modest positive deviations, indicating consistent market premiums for higher responsibility and experience. At the other end, junior, mid-level, and lead positions systematically underpay relative to expectations, suggesting tighter salary bands and lower negotiation leverage for early-career roles. Supervisor roles fall slightly below neutrality, reflecting their operational rather than strategic nature in most organisations. Overall, the pattern is clear: salary deviations align directly with responsibility level, with principal > manager > senior > assistant/executive ≈ neutral > lead/mid/junior in terms of adjusted salary premiums.",
            "job_title": "Job title residuals summarise the most granular persistent premiums/penalties after controls. The weighted fairness results show a clear structural hierarchy across job families: data scientists tend to be paid above model expectations, data engineers cluster near neutral to mildly negative values, and data analysts consistently fall below predicted norms. Roles explicitly tied to ML/AI, especially in data science tracks, show strong positive deviations, reflecting the premium placed on advanced modelling and AI-related skills. More specialised scientist roles (health, research, security) also trend positive, suggesting that domain-specialised expertise carries additional salary value even after controlling for skills and seniority. In contrast, analyst roles—regardless of domain—systematically underpay relative to expectation, likely reflecting lower technical depth and weaker bargaining power in the market. Overall, the job family fairness analysis exposes a strong stratification: ML/AI data scientists > general data scientists > domain scientists > engineers > analysts in terms of adjusted salary premiums.",
        }
        _LABELS = {
            "location": "Location (state)",
            "sector": "Sector",
            "company_size": "Company size",
            "ownership": "Ownership type",
            "seniority": "Seniority",
            "job_title": "Job title (title_rich)",
        }

        options = [k for k in _LABELS.keys() if k in group_types]
        if not options:
            st.error(f"No recognised group types in assets. Found: {group_types}")
            return

        gt = st.selectbox(
            "Group by", options, format_func=lambda k: _LABELS.get(k, k), index=0
        )

        metric = st.selectbox(
            "Sort/plot by",
            ["mean_residual", "size_weighted_mean", "median_residual", "n"],
            index=1,
        )

        df = (
            group.loc[group["group_type"] == gt]
            .copy()
            .sort_values(metric, ascending=False)
        )

        _plot_group_bars(df, metric=metric, title=f"{_LABELS.get(gt, gt)} — {metric}")
        st.caption("Feature-level fairness insights.")

        _interp_expander(
            f"Interpretation — {_LABELS.get(gt, gt)}",
            _INTERPRETATION.get(gt, "Interpretation pending."),
            expanded=False,
        )

        st.markdown("### Table")
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    # -----------------------------
    # SHAP importance
    # -----------------------------
    st.subheader("SHAP explainability")

    _interp_expander("SHAP primer", _SHAP_INTRO, expanded=False)

    try:
        shap_values = _load_shap_explanation()
    except Exception as e:
        st.error(f"Could not load SHAP artefact: {e}")
        return

    st.markdown("### Global SHAP")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        fig_bar = _global_shap_bar_fig(shap_values)
        st.pyplot(fig_bar, clear_figure=True)

    with c2:
        fig_bee = _beeswarm_fig(shap_values, max_display=20)
        st.pyplot(fig_bee, clear_figure=True)

    _interp_expander(
        "Interpretation — Global SHAP and Beeswarn", _SHAP_GLOBAL_INTERP, expanded=False
    )

    st.divider()
    st.markdown("### Local SHAP (top categorical drivers)")

    lookups = _load_ch1_lookup_tables()

    local_opts = {
        "state_code": ("state_code", "state", lookups["state"]),
        "sector_code": ("sector_code", "Sector", lookups["sector"]),
        "title_rich_code": ("title_rich_code", "title_rich", lookups["title"]),
    }

    choice = st.selectbox(
        "Choose a categorical feature to inspect",
        ["state_code", "sector_code", "title_rich_code"],
        index=0,
        format_func=lambda x: {
            "state_code": "State (state_code)",
            "sector_code": "Sector (sector_code)",
            "title_rich_code": "Job title (title_rich_code)",
        }[x],
    )

    code_col, label_col, lookup_df = local_opts[choice]

    # Fixed (no slider) to keep the UI clean and consistent.
    top_n = 12 if choice == "title_rich_code" else 30

    try:
        df_local = _local_shap_frames(
            shap_values,
            feature=choice,
            code_col=code_col,
            label_col=label_col,
            lookup=lookup_df.rename(
                columns={
                    lookup_df.columns[0]: code_col,
                    lookup_df.columns[1]: label_col,
                }
            ),
        )
        fig_local = _plot_local_bar_and_box(
            df_local, label_col=label_col, title=choice, top_n=top_n
        )
        st.pyplot(fig_local, clear_figure=True)
        st.caption("Showing absolute local mean SHAP.")
    except Exception as e:
        st.error(f"Could not plot local SHAP for {choice}: {e}")
        return

    _interp_expander(
        "Interpretation — Local SHAP",
        _LOCAL_INTERP.get(choice, "Interpretation pending."),
        expanded=False,
    )
