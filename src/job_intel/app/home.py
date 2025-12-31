# src/job_intel/app/home.py

from __future__ import annotations

from pathlib import Path

import streamlit as st


def _repo_root() -> Path:
    # .../src/job_intel/app/home.py -> repo root = parents[3]
    return Path(__file__).resolve().parents[3]


def _resolve_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


def _figure_path(filename: str, *, mac_abs: str | None = None) -> Path | None:
    repo = _repo_root()

    candidates: list[Path] = []

    # 1) User-provided absolute path (works locally)
    if mac_abs:
        candidates.append(Path(mac_abs))

    # 2) Repo-relative (works on other machines / deployments)
    candidates.append(repo / "docs" / "narrative" / "figures" / filename)
    candidates.append(
        repo / "src" / "job_intel" / "docs" / "narrative" / "figures" / filename
    )

    return _resolve_first_existing(candidates)


def render() -> None:
    st.title("Job Intelligence Engine")
    st.caption(
        "A decision-support app that turns job-ad data into market insights, personalised recommendations, and an upskilling plan."
    )

    engine_path = _figure_path(
        "engine_path.png",
        mac_abs="/Users/alejandrofp/Desktop/Projects/03_Flagship_Portfolio/job-intelligence-engine/docs/narrative/figures/engine_path.png",
    )

    # 50/50 split so the figure is readable
    left, right = st.columns([1, 1], vertical_alignment="top")
    with left:
        st.markdown(
            """
Most people don’t struggle to work hard — they struggle to **choose**: which jobs to target, what “good fit” means in practice,
and which skills actually change outcomes (instead of just adding noise).

Job Intelligence Engine turns the **data** job market into something you can **query**: it summarises the market signal, then maps your current
skills to **best-now roles**, **stretch roles**, and an **ROI-ranked upskilling plan** that’s grounded in real job-posting patterns.


This project addresses a common problem: job search is noisy and time-consuming, and it’s hard to know which roles are realistic now,
which are worth stretching for, and which skills will move the needle most.

This app uses a unified pipeline built on **data-related** job-ad skill signals and a salary model to help you make **clear, evidence-based decisions**.
""".strip()
        )
    with right:
        if engine_path is not None:
            st.image(engine_path.as_posix(), use_container_width=True)
        else:
            st.info("Figure missing: docs/narrative/figures/engine_path.png")

    st.divider()

    st.subheader("What you can do here")
    st.markdown(
        """
- **Landscape:** understand the global data job market signal (salary residuals, fairness lenses, and skill value ranking).
- **Recommender:** enter your constraints and skills to get two job buckets:
  - **Best now** (low barrier / high fit)
  - **Stretch** (higher barrier, but strong upside)
- **Upskilling:** get an ROI-ranked upskilling plan and “co-learning” suggestions (skills that tend to appear together across the market).
""".strip()
    )

    st.divider()

    simple_workflow = _figure_path(
        "simple_workflow.png",
        mac_abs="/Users/alejandrofp/Desktop/Projects/03_Flagship_Portfolio/job-intelligence-engine/docs/narrative/figures/simple_workflow.png",
    )

    # 50/50 split so the figure is readable
    left2, right2 = st.columns([1, 1], vertical_alignment="top")
    with left2:
        st.subheader("How to use the app (recommended flow)")
        st.markdown(
            """
1. **Start with Landscape**  
   Get context on the market signal so the recommendations feel interpretable.

2. **Run the Recommender**  
   Provide your **skills** (free text) and optional constraints (state, sector, salary target).  
   The output includes explained ranking and bucket logic, job descriptions, and a small glossary.

3. **Explore Upskilling**  
   After you run the recommender, this page proposes the best skill families to learn next (high ROI, low downside),
   plus co-learning neighbours from the market embedding space.
""".strip()
        )
    with right2:
        if simple_workflow is not None:
            st.image(simple_workflow.as_posix(), use_container_width=True)
        else:
            st.info("Figure missing: docs/narrative/figures/simple_workflow.png")

    st.divider()

    st.subheader("Inputs (what to type)")
    st.markdown(
        """
- **skill_text:** paste your current skills, tools, and experience in plain language.
- **current_state:** e.g., `CA` or `ALL`.
- Optional: **target sectors**, **job family/title**, **salary target**.
""".strip()
    )

    st.divider()

    st.subheader("Important notes")
    st.markdown(
        """
- Outputs are **decision support**, not guarantees.
- Market and skill signals are **descriptive**, not causal.
- The goal is to make trade-offs explicit: fit vs barrier vs salary potential, and which skills improve positioning with minimal downside.
""".strip()
    )

    st.info(
        "Use the sidebar navigation to open **Landscape**, **Recommender**, and **Upskilling**."
    )

    st.divider()
    st.subheader("Want to learn more?")
    st.link_button(
        "GitHub repository (full code + documentation)",
        "https://github.com/AlejandroFuentePinero/job-intelligence-engine",
        type="secondary",
    )
