from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.job_intel.config import PROCESSED_DATA_DIR


def _make_hist_bins(x: pd.Series, bins: int = 40) -> pd.DataFrame:
    x = pd.to_numeric(x, errors="coerce").dropna()
    counts, edges = np.histogram(x.to_numpy(), bins=bins)
    out = pd.DataFrame(
        {"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts}
    )
    out["bin_mid"] = (out["bin_left"] + out["bin_right"]) / 2.0
    return out


def _make_box_stats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    whisker_low = float(np.min(x[x >= lo])) if np.any(x >= lo) else float(np.min(x))
    whisker_high = float(np.max(x[x <= hi])) if np.any(x <= hi) else float(np.max(x))
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "q1": float(q1),
        "median": float(med),
        "q3": float(q3),
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
    }


def _group_summary(df: pd.DataFrame, col: str, group_type: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["residuals"] = pd.to_numeric(tmp["residuals"], errors="coerce")

    g = (
        tmp.groupby(col, dropna=False)["residuals"]
        .agg(
            n="count",
            mean_residual="mean",
            median_residual="median",
            p10=lambda s: (
                float(np.nanpercentile(s.dropna(), 10)) if s.notna().any() else np.nan
            ),
            p90=lambda s: (
                float(np.nanpercentile(s.dropna(), 90)) if s.notna().any() else np.nan
            ),
        )
        .reset_index()
        .rename(columns={col: "group_value"})
    )

    total_n = float(g["n"].sum()) if g["n"].sum() else 0.0
    g["size_weighted_mean"] = (
        g["mean_residual"] * (g["n"] / total_n) if total_n > 0 else 0.0
    )
    g.insert(0, "group_type", group_type)
    return g


def build_fairness_assets(
    *,
    in_path: Path | None = None,
    out_dir: Path | None = None,
    bins: int = 30,
) -> dict[str, Path]:
    in_path = in_path or (PROCESSED_DATA_DIR / "df_with_residuals.csv")
    out_dir = out_dir or (PROCESSED_DATA_DIR / "ch5_assets")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    if "residuals" not in df.columns:
        raise ValueError("df_with_residuals.csv must contain a 'residuals' column.")

    # overall distro
    hist = _make_hist_bins(df["residuals"], bins=bins)
    (out_dir / "fairness_residual_hist_bins.csv").write_text(hist.to_csv(index=False))

    box_stats = _make_box_stats(df["residuals"])
    (out_dir / "fairness_residual_box_stats.json").write_text(
        json.dumps(box_stats, indent=2)
    )

    # ONLY the 6 groupings you listed
    group_cols = [
        ("state", "location"),
        ("Sector", "sector"),
        ("Size", "company_size"),
        ("ownership_clean", "ownership"),
        ("seniority_combined", "seniority"),
        ("title_rich", "job_title"),
    ]

    missing = [c for c, _ in group_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in df_with_residuals.csv: {missing}"
        )

    fairness_long = pd.concat(
        [_group_summary(df, col=c, group_type=t) for c, t in group_cols],
        ignore_index=True,
    )

    fairness_path = out_dir / "fairness_group_summary_long.csv"
    fairness_long.to_csv(fairness_path, index=False)

    return {
        "hist_bins": out_dir / "fairness_residual_hist_bins.csv",
        "box_stats": out_dir / "fairness_residual_box_stats.json",
        "group_summary_long": fairness_path,
    }


if __name__ == "__main__":
    paths = build_fairness_assets()
    for k, p in paths.items():
        print(f"{k}: {p}")
