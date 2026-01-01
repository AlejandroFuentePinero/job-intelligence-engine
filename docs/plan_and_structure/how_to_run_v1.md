# Job Intelligence Engine — How to Run (v1)

This repo ships a deterministic, end-to-end **Job Intelligence Engine** and a lightweight **Streamlit app** (`app.py`) that surfaces:
- **Market landscape** (fairness residuals, global SHAP, global skill value ranking)
- **Recommender** (`best_now` / `stretch`) + job-level “why”
- **Upskilling** (counterfactual deltas) + **macro co-learning** (skill neighbour view)

The app is designed to run **without training** (models are loaded from `models/`; assets are assembled/validated by Chapter 5 build scripts).

---

## 1) Prerequisites

- Python **3.10+** (recommended: 3.11)
- macOS / Linux / Windows (WSL recommended on Windows)
- `pip` available (or `uv`/`pipx` if you prefer; instructions below use `pip`)

---

## 2) Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate   # Windows (PowerShell)

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3) One-command run (build + launch)

From the repository root:

```bash
python -m src.job_intel.pipelines.ch5_app_build && streamlit run app.py
```

What this does:
1) **Builds** (or refreshes) app-ready Chapter 5 artefacts (currently fairness assets) and  
2) **Validates** that all required runtime artefacts exist on disk (fail-fast if anything is missing), then  
3) Launches the Streamlit app via the root entrypoint `app.py`.

---

## 4) What you should see in the app

1) Open the **Home** page and read the “how to use” flow (landscape → recommender → upskilling).
2) Go to **Landscape** to understand the market mechanics.
3) Go to **Recommender**:
   - click **Load demo persona**
   - click **Run recommender**
4) Go to **Upskilling + Macro** to view top upskill families and co-learning neighbours.

---

## 5) Data and artefacts policy (important)

This project is designed as a portfolio-quality codebase with deterministic pipelines.

### What is typically committed
- `src/` code (all pipelines/features/models/app pages)
- **trained model artefacts** under `models/` (salary model + PCA + 27 skill models), if you choose to ship them
- small **demo configs** under `src/job_intel/evaluation/` (e.g., `recommender_demo.json`)
- lightweight **app assets** under `data/processed/ch5_assets/` *if you choose to ship them*

### What is typically NOT committed
- Large raw datasets (`data/raw/`)
- Large intermediate processed outputs (`data/processed/`), unless explicitly needed for demo

Your `.gitignore` and README should make this explicit. If artefacts are not committed, users must (re)generate them using the pipelines below.

---

## 6) Smoke test (shipping check)

To run a lightweight end-to-end check (no UI automation):

```bash
python -m src.job_intel.evaluation.ch5_smoke_test
```

This:
- runs the Chapter 5 build/validation step
- loads the demo persona config
- executes the Chapter 4 pipeline in “app mode”
- verifies output contracts (tables exist, job_ids present, non-empty buckets)

Optional determinism check (if implemented in your file):
```bash
python -m src.job_intel.evaluation.ch5_smoke_test --check-determinism
```

---

## 7) Full rebuild (from raw data) — optional

If you want to reproduce *everything* from scratch (including processed tables), you’ll need the raw dataset.

1) Download the raw data (e.g., Kaggle source referenced in README)
2) Place raw files into:
```
data/raw/
```

Then run, in order:

```bash
python -m src.job_intel.pipelines.chapter0_build_base_dataset
python -m src.job_intel.pipelines.chapter1_models --which salary
python -m src.job_intel.pipelines.chapter1_models --which skills
python -m src.job_intel.pipelines.chapter2_hidden_structures
python -m src.job_intel.pipelines.ch5_app_build
streamlit run app.py
```

Notes:
- Chapter 1 training can take time depending on hardware.
- Chapter 2 embedding steps can be compute-heavy; they are *not required* for the v1 app beyond the skill-neighbour artefact (if you already ship it).

---

## 8) Troubleshooting

### “Missing artefact” error
Run the build/validator explicitly:
```bash
python -m src.job_intel.pipelines.ch5_app_build
```
It will print which expected files are missing.

### App runs but plots are empty
Verify the Chapter 5 assets exist:
- `data/processed/ch5_assets/`
- and the required model files exist under `models/`

### Streamlit import / module path issues
Run commands from the **repo root** (same directory as `app.py`).

---

## 9) Minimal run checklist (for interviewers)

```bash
pip install -r requirements.txt
python -m src.job_intel.pipelines.ch5_app_build
streamlit run app.py
```
