# Job Intelligence Engine – Project Structure & Workflow Guide  
*(Contributor Onboarding & Development Manual — Version 2)*

This document explains **everything an user needs to know** to work inside this project correctly and confidently.  
It is explicit, practical, and written so that no prior experience with the project layouts is required.

---

# 1. Core Concept: The Layered Workflow

The entire repository is built around a clean DS/ML engineering workflow:

```
Exploration → Production Code → Pipelines → Outputs → Documentation
```

Each stage has its own folder and its own rules.  
Understanding this flow makes the project easy to navigate and contribute to.

---

# 2. Repository Layout

```
job-intelligence-engine/
│
├── data/
│   ├── raw/            # untouched downloaded data
│   ├── interim/        # temporary intermediate files
│   ├── processed/      # clean, model-ready datasets
│   └── external/       # external resources (O*NET, skill lists)
│
├── docs/               # all documentation (text, diagrams, decisions)
│
├── notebooks/          # exploration + prototyping only
│
├── reports/            # final figures, tables, written outputs
│
├── models/             # saved trained models
│
├── src/
│   └── job_intel/
│       ├── app/        # dashboards / API / front-end (optional)
│       ├── data/       # data IO and schema enforcement
│       ├── features/   # feature engineering (skills, titles, embeddings)
│       ├── modelling/  # ML model training & inference (British spelling)
│       ├── pipelines/  # end-to-end workflows combining modules
│       ├── evaluation/ # metrics + validation logic
│       ├── utils/      # logging, timing, helpers, decorators
│       ├── config.py   # global paths, constants
│       └── __init__.py
│
├── tests/              # unit tests for functions inside src/
│
└── README.md
```

Everything else is either an input, an output, or documentation.

---

# 3. The Contribution Philosophy (How to Work)

Every new feature or analysis follows this cycle:

```
1. Explore in a notebook
2. Extract stable logic into src/
3. Build/update a pipeline in pipelines/
4. Save outputs to data/processed, models/, and reports/
5. Update documentation in docs/
```

Short rules:

- **Notebooks are scratchpads**, not production code.  
- **src/** is the source of truth for all reusable logic.  
- **pipelines/** orchestrate modules into workflows.  
- **data/** holds all data.  
- **docs/** records decisions & architecture.  
- **reports/** holds final outputs.

If you follow this loop, you will always know what to do next.

---

# 4. Folder Responsibilities (Detailed)

Below is the precise purpose of each folder so contributors know exactly where to put things.

---

## 4.1. `data/` — All Project Data

| Folder        | Purpose                                                                 |
|---------------|-------------------------------------------------------------------------|
| `raw/`        | Original downloaded files (never edited).                               |
| `interim/`    | Temporary outputs during cleaning.                                      |
| `processed/`  | Final clean datasets ready for modelling & pipelines.                   |
| `external/`   | Third-party reference datasets (O*NET, skill dictionary, etc.).         |

**Rules:**
- Never put data inside notebooks or src.  
- Do not modify raw data—always clean into interim or processed.  
- Pipelines are responsible for writing final datasets.

---

## 4.2. `notebooks/` — Exploration & Prototyping

Use this folder to:

- perform EDA  
- inspect raw data  
- try new models or features  
- visualise results  
- write quick experiments  
- sketch logic before formalising it  

Do *not* put:

- reusable functions  
- model training code  
- data IO functions  
- pipelines  
- final outputs  

**Rule of thumb:**  
Once a cell contains logic you want to reuse → move that logic to `src/job_intel/...`.

---

## 4.3. `src/job_intel/` — Production Code (The Engine)

This is where the real system lives.  
Each subfolder has a specific role:

---

### 4.3.1 `data/` — Input/Output + Schema Handling

Purpose:

- consistent loading and saving of datasets  
- schema validation (required columns, types)  
- minimal cleaning operations close to IO  

Examples:

```python
def load_raw_jobs(path): ...
def save_processed_jobs(df, path): ...
def validate_job_schema(df): ...
```

All other modules should rely on these functions instead of custom IO.

---

### 4.3.2. `features/` — Feature Engineering

Includes:

- job title normalisation  
- skill extraction  
- multi-hot skill vectors  
- Node2Vec/graph features  
- job embeddings  
- clustering (job families, skill bundles)  

Features transform clean data into modelling-ready inputs.

Examples:

```
features/title_normaliser.py
features/skill_parser.py
features/job_skill_graph.py
```

---

### 4.3.3. `modelling/` — Model Training & Inference 

Handles:

- salary models (XGBoost, LightGBM, etc.)  
- skill requirement models  
- explainability modules (SHAP, PDP, ICE)  
- embedding models  
- prediction wrappers  
- model save/load utilities  

Each model module should follow a simple structure:

```python
def train(df, params): ...
def predict(model, df): ...
def save(model, path): ...
def load(path): ...
```

---

### 4.3.4. `pipelines/` — End-to-End Workflows

This is one of the **most important** additions.

Pipelines combine modules from:

- `data/`
- `features/`
- `modelling/`
- `evaluation/`

into complete reproducible workflows.

Examples:

```
pipelines/title_normalisation_pipeline.py
pipelines/skill_extraction_pipeline.py
pipelines/salary_training_pipeline.py
pipelines/recommendation_pipeline.py
```

A typical pipeline:

```python
def run(config):
    df = load_processed_jobs(config["input"])
    df_features = build_features(df)
    model = train(df_features, config["model_params"])
    save(model, config["model_output"])
```

Pipelines allow you to reproduce results without relying on notebooks.

---

### 4.3.5. `evaluation/` — Metrics & Validation

Contains:

- regression/classification metrics  
- cross-validation wrappers  
- diagnostic plots  
- fairness/residual analysis tools  

Should not perform data IO or modelling itself.

---

### 4.3.6. `utils/` — Helper Utilities

Small functions used across the project:

- logging  
- timing decorators  
- file/path helpers  
- string cleaning helpers  
- reusable decorators  
- universal plotting formatting  

This prevents code duplication across modules.

---

### 4.3.7. `app/` — Application Layer (Optional)

If you eventually build:

- a dashboard (Streamlit, Dash, etc.)  
- a REST API  
- a command-line interface  

…it goes here.

The app should call pipelines, not raw functions.

---

## 4.4. `models/` — Trained Model Artefacts

Pipelines save trained models here:

- `.joblib`  
- `.pkl`  
- `.bin` (embeddings)  

Each file should be versioned:

```
salary_model_v1.joblib
skill_req_model_v2.pkl
node2vec_embeddings_v1.bin
```

---

## 4.5. `reports/` — Final Outputs for Humans

Contains:

- figures  
- tables  
- “insight summaries”  
- Markdown reports (like `project_report.md`)  

Anything intended for communication lives here.

---

## 4.6. `docs/` — The Project Brain

Contains documentation that explains:

- what the project is (`overview.md`)  
- the conceptual plan (`project_plan.md`)  
- the system architecture (`architecture.md`)  
- the data structures (`data_dictionary.md`)  
- the workflow & project structure (this file)  

Whenever you add new functionality, update the relevant document.

---

# 5. How to Develop a New Feature (Step-by-Step)

This is the exact process every contributor should follow.

---

## Step 1 — Start in a Notebook

Use a notebook to:

- load data  
- explore patterns  
- try algorithms  
- test logic  
- visualise outputs  

Notebook code can be rough.

---

## Step 2 — Move Stable Logic into `src/job_intel/`

Once you’re confident that logic is:

- correct  
- reusable  
- important  

Move it from the notebook into the appropriate module.

After moving it:

```python
from src.job_intel.features.skill_parser import extract_skills
```

No copy-paste duplication — the notebook imports the function.

---

## Step 3 — Build or Update a Pipeline

Every multi-step process should become a pipeline.

Example:

```
run_salary_training_pipeline.py
```

A pipeline:

- loads clean data  
- builds features  
- trains models  
- evaluates performance  
- saves results  
- writes outputs  

This guarantees reproducibility.

---

## Step 4 — Save Outputs

Depending on what you built, save outputs to:

| Type | Folder |
|------|--------|
| Clean data | `data/processed/` |
| Intermediate temporary | `data/interim/` |
| Trained model | `models/` |
| Plots, tables | `reports/` |

Never store outputs in notebooks or src.

---

## Step 5 — Update Documentation

Update:

- `architecture.md` → describe new component  
- `data_dictionary.md` → add new columns/features  
- `project_plan.md` → if project direction changed  
- `workflow.md` → only if conventions changed  

Documentation is part of the workflow.

---

# 6. Example Contribution: Skill Value Analysis

1. **Notebook:**  
   Prototype salary uplift + rarity methods.

2. **Features:**  
   Move uplift & rarity functions to  
   `src/job_intel/features/skill_value.py`.

3. **Pipeline:**  
   Create  
   `src/job_intel/pipelines/skill_value_pipeline.py`.

4. **Outputs:**  
   Save:

   - `data/processed/skill_value_scores.parquet`  
   - `reports/figures/top_skill_values.png`

5. **Documentation:**  
   Update:

   - architecture.md  
   - data_dictionary.md  

6. **Tests:**  
   Add `tests/test_skill_value.py`.

---

# 7. Mental Model (ASCII Diagram)

```
              NOTEBOOKS
 (exploration, EDA, prototype logic)
                  │
                  ▼
     ┌─────────────────────────────┐
     │        SRC/JOB_INTEL        │
     │ data/ features/ modelling/  │
     │ pipelines/ evaluation/ utils│
     └─────────────────────────────┘
                  │
                  ▼
         END-TO-END PIPELINES
                  │
 ┌────────────────┼──────────────────┐
 ▼                ▼                  ▼
data/processed/  models/          reports/
(clean data)   (trained models) (figures + tables)
 └─────────────────────────────────────────────┘
                  │
                  ▼
                DOCS
     (architecture, plan, workflow, dictionary)
```

---

# 8. Key Rules (Cheat Sheet)

- **All real logic lives in `src/job_intel/`**  
- **Notebooks are for exploration only**  
- **Pipelines control end-to-end execution**  
- **Data never lives in src or notebooks**  
- **Model files never live in notebooks**  
- **Documentation evolves with the code**  
- **British spelling: `modelling/`**  
- **Consistency is more important than cleverness**

---

This document describes exactly how to work inside the project with clarity and structure.  
Follow it, and the entire system will stay clean, reproducible, and easy for others to build upon.

