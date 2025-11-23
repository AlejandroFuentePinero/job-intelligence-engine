# Job Intelligence Engine – Project Structure & Workflow Guide

This document defines:

1. How the project is organised  
2. Where every file type belongs  
3. How notebooks and src code interact  
4. The expected development workflow  
5. Rules that keep the project scalable and contributor-friendly

Any collaborator should be able to read this and start working immediately.  
This also serves as a reference for “future you”.

---

# 1. Top-Level Layout

```
job-intelligence-engine/
│
├── data/
│   ├── raw/            # untouched datasets (downloaded exactly as received)
│   ├── interim/        # intermediate staging outputs
│   ├── processed/      # clean, model-ready analytical tables
│   └── external/       # third-party reference datasets (e.g., O*NET)
│
├── docs/               # project documentation
│   ├── overview.md
│   ├── project_plan.md
│   ├── architecture.md
│   ├── data_dictionary.md
│   └── project_structure_and_workflow.md
│
├── notebooks/          # exploration and prototyping only
│   ├── 01_eda_*.ipynb
│   ├── 02_title_normalization.ipynb
│   ├── 03_skill_extraction.ipynb
│   ├── 04_salary_model.ipynb
│   └── ...
│
├── reports/            # final visual and written outputs
│   ├── figures/
│   ├── tables/
│   └── project_report.md
│
├── models/             # trained models (.pkl, .joblib)
│
├── src/
│   └── job_intel/
│       ├── app/        # dashboards or front-end logic
│       ├── data/       # data loading, saving, schemas, transformations
│       ├── evaluation/ # evaluation metrics and diagnostics
│       ├── features/   # feature engineering: titles, skills, embeddings
│       ├── models/     # salary models, skill models, inference code
│       ├── __init__.py
│       ├── config.py   # paths, constants, global configuration
│       └── test_black.py
│
├── tests/              # unit tests for src functions
│
├── README.md
├── requirements.txt
└── .env
```

---

# 2. What Goes Where (By File Type)

## 2.1. Data Files

| Folder            | Content Description                                    | Notes                        |
|-------------------|--------------------------------------------------------|-------------------------------|
| `data/raw/`       | Original source files exactly as downloaded            | Never modified               |
| `data/interim/`   | Stepwise processed files                              | Optional staging             |
| `data/processed/` | Clean, model-ready data tables                         | Created by code/notebooks    |
| `data/external/`  | Reference datasets (e.g., skill dictionaries, O*NET)   | Not generated in project     |

---

## 2.2. Notebooks (`notebooks/`)

Used for:
- Exploration  
- EDA  
- Trying out ideas  
- Small tests and manual inspections  

Notebooks **do not** contain:
- Production pipelines  
- Full training logic  
- Reusable functions  

Once logic is stable → migrate it to `src/job_intel/`.

---

## 2.3. Source Code (`src/job_intel/`)

This folder contains **production-quality, reusable code**.

### Subfolders:

| Folder              | Purpose                                           |
|---------------------|---------------------------------------------------|
| `data/`             | Data IO, schema checks, table transformations     |
| `features/`         | Title normalization, skill extraction, embeddings |
| `models/`           | Salary models, skill models, prediction pipelines |
| `evaluation/`       | Metrics, diagnostics, validation utilities        |
| `app/`              | Dashboards, API endpoints (if any)                |
| `config.py`         | Paths, constants, global settings                 |

Workflow:  
Notebooks → functions → `.py` modules → imported back into notebooks.

---

# 3. Development Workflow

A typical cycle:

1. Explore something new in a notebook  
2. Prototype logic  
3. Test with small data  
4. Move stable logic → `src/job_intel/...`  
5. Import functions back into notebook  
6. Use notebook for visualizations + interpretation  
7. Save final clean outputs → `data/processed/`  
8. Document any changes in `docs/`  

---

# 4. Rules for Clean Workflow

### Rule 1 — One Source of Truth  
All real logic lives in `src/job_intel/`.

### Rule 2 — Notebooks Are Scratchpads  
Messy exploration is allowed because the final code lives in `src/`.

### Rule 3 — Data Is Versioned  
Raw → interim → processed.

### Rule 4 — Document As You Go  
Use the `docs/` folder for architecture notes, data dictionary, and design logs.

### Rule 5 — Don’t Over-Polish Early  
Move forward, refine once components stabilize.

---

# 5. How Contributors Should Work

This document + `architecture.md` is the onboarding package.  
A contributor should understand immediately:

- where to put new code  
- how to update data  
- how the pipeline flows  
- how notebooks and src interact  

---

# 6. Example of a Normal Work Session

1. Open a notebook (e.g. `03_skill_extraction.ipynb`)  
2. Try small experiments  
3. Prototype logic  
4. Move stable code to `src/job_intel/features/skill_extraction.py`  
5. Replace notebook code with an import  
6. Save cleaned data to `data/processed/`  
7. Update docs if needed  
8. Done

---

# 7. Mental Model of the Entire Project

The project is a **pipeline**:

- Explore in notebooks  
- Build stable code in `src/`  
- Save outputs in `data/processed/`  
- Interpret in `reports/`  
- Explain in `docs/`  

This structure prevents chaos and makes the project scalable and maintainable.
