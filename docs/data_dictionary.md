# Job Intelligence Engine — Data Dictionary & Overview

## 1. Introduction

This document describes the **final processed dataset** produced in **Chapter 0** of the *Job Intelligence Engine*. It includes the data sources, preprocessing pipeline, transformations applied, column definitions, missing‑data decisions, and modelling readiness.  
This dataset is now the **canonical foundation** for all subsequent chapters.

---

## 2. Dataset Summary

| Item | Value |
|------|-------|
| **Rows** | 6,162 |
| **Columns** | 47 |
| **Source Files** | Data Analyst CSV + Data Scientist CSV (Kaggle / Glassdoor) |
| **Final Format** | CSV |
| **Pipeline Script** | `src/job_intel/pipelines/chapter0_build_base_dataset.py` |

---

## 3. Provenance & Scope

Two raw CSVs (“Data Analyst” and “Data Scientist” Glassdoor job listings) were ingested, cleaned, merged, enriched, and transformed into a modelling‑ready dataset.  
All preprocessing is fully reproducible through the pipeline.

**Primary goals of Chapter 0:**

1. Clean & normalize raw job data  
2. Standardize job titles & extract seniority  
3. Extract skills using a custom taxonomy  
4. Parse salary ranges  
5. Add domain labels using SBERT+KMeans lookup  
6. Engineer company metadata (ownership, size, location)  
7. Handle missing values  
8. Prune noisy/unusable fields  
9. Produce a single, stable master dataset

The output is **analysis-grade** and safe for all downstream modelling.

---

## 4. Pipeline Overview

### Step-by-step

1. **Load Raw Files**  
   - Align columns  
   - Replace Glassdoor placeholder -1 with NA  
   - Add `role_source` (DA or DS)

2. **Minor Feature Cleaning**  
   - Extract `state` from Location  
   - Extract `state_hq` (later dropped)  
   - Harmonize foreign states → `"international"`  
   - Create `ownership_clean`  
   - Convert `Founded` → Int64  
   - Drop ultra‑sparse fields (Competitors, Easy Apply, Revenue, etc.)

3. **Text Cleaning**  
   - Create `job_description_clean`  
   - Light normalization (keeps meaning)

4. **Title Processing & Seniority Extraction**  
   - Clean raw title (`job_title_base`)  
   - Extract Roman numerals  
   - Extract seniority from title  
   - Extract seniority from description  
   - Combine → `seniority_combined`  
   - Build normalized job family (`job_title_family`)  
   - Domain assignment via SBERT lookup (`domain`)

5. **Salary Parsing**  
   - Parse raw salary string  
   - Standardize hourly → annual  
   - Extract `sal_min`, `sal_max`, `sal_mean`

6. **Skill Extraction**  
   - Combine title + description into `title_plus_description`  
   - Dictionary-based extraction from custom taxonomy  
   - Produce 30+ multi-hot skill indicators across 10 categories  
   - Ensures each is strictly 0/1

7. **Missing Data Handling**  
   - Drop rows missing both *description* and *salary*  
   - Fill `Sector`, `Industry`, `Size` with `"Unknown"`  
   - Keep missing values in `Rating` and `Founded` (useful signal)

8. **Column Pruning**  
   Drop raw fields:  
   - Raw title  
   - Seniority intermediate fields  
   - Salary raw text  
   - `state_hq`  
   - Skill extraction intermediates  
   - All columns retained were explicitly chosen via `keep` list

---

## 5. Final Column Dictionary (47 Columns)

### A. Company Metadata

| Column | Meaning |
|--------|---------|
| `Rating` | Glassdoor rating (0–5). Missing preserved. |
| `Size` | Company size category (filled to "Unknown" if missing). |
| `Founded` | Year founded (Int64). |
| `Industry` | Company industry (filled to "Unknown"). |
| `Sector` | Company sector (filled to "Unknown"). |
| `ownership_clean` | Standardized: `private`, `public`, `nonprofit`, `government`, `unknown`. |
| `state` | State abbreviation or `"international"` for non-US roles. |

---

### B. Role Identity

| Column | Meaning |
|--------|---------|
| `seniority_combined` | Unified seniority extracted from title + description. |
| `job_title_family` | Normalized family label (analyst/scientist/engineer/etc.). |
| `domain` | SBERT/KMeans-based domain cluster. |
| `role_source` | `data_analyst` or `data_scientist` origin dataset. |

---

### C. Salary Features

| Column | Meaning |
|--------|---------|
| `sal_min` | Minimum annualized salary. |
| `sal_max` | Maximum annualized salary. |
| `sal_mean` | Mean of min/max. |

---

### D. Skills (Multi-hot Flags)

Each field is **0 or 1**, meaning **at least one matching token** was found in the text.

#### Core Programming
- `core_programming__basic`
- `core_programming__intermediate`
- `core_programming__advanced`

#### Data Engineering & Pipelines
- `data_engineering_pipelines__basic`
- `data_engineering_pipelines__intermediate`
- `data_engineering_pipelines__advanced`

#### Machine Learning & AI
- `ml_ai__basic`
- `ml_ai__intermediate`
- `ml_ai__advanced`

#### Analytics & Statistics
- `analytics_stats__basic`
- `analytics_stats__intermediate`
- `analytics_stats__advanced`

#### BI & Visualisation
- `bi_viz__basic`
- `bi_viz__intermediate`
- `bi_viz__advanced`

#### Cloud / MLOps
- `cloud__basic`
- `cloud__intermediate`
- `cloud__advanced`

#### Databases & Storage
- `db_storage__basic`
- `db_storage__intermediate`
- `db_storage__advanced`

#### Productivity & Workflow
- `productivity_workflow__basic`
- `productivity_workflow__intermediate`
- `productivity_workflow__advanced`

#### Soft Skills
- `soft_skills__core`
- `soft_skills__leadership`

#### Domain-Specific
- `domain_specific__none`

---

### E. Verification Columns (Internal Consistency)

| Column | Meaning |
|--------|---------|
| `title_plus_description` | Text used for skill extraction. |
| `Job Description` | Raw description (kept for inspection). |
| `job_title_base` | Cleaned job title after noise removal. |
| `job_title_norm` | Normalized intermediate title tokens. |
| `job_description_clean` | Clean text used in seniority+skills. |

---

## 6. Missing Value Policy

| Variable | Treatment |
|----------|-----------|
| Description + Salary BOTH missing | Row dropped |
| Industry / Sector / Size | Filled `"Unknown"` |
| Rating | Keep NA |
| Founded | Keep NA |
| Salary fields | NA preserved |
| Skill flags | Always 0/1 |

This approach preserves meaningful uncertainty for modelling.

---

## 7. Modelling Readiness

This dataset is **ready for**:

- Salary prediction (regression)  
- Seniority classification  
- Job family/domain classification  
- Skill demand modelling  
- Clustering, embeddings, topic modelling  
- Recommendation systems  
- Graph skill-job networks  

### Strengths:
- Clean structured features  
- Multi-hot skill encoding  
- High-quality salary parsing  
- Domain mapping via SBERT  
- Seniority extraction from two sources  
- Full transparency (raw + cleaned text kept)

---

## 8. Limitations

Expected for real-world scraped data:

- Some `Rating` and `Founded` values missing  
- Industry labels noisy  
- Skill taxonomy v1 may expand in future  
- Seniority extraction is heuristic-based  
- Description cleaning intentionally light (NLP enhancements possible)

These do not hinder modelling but define future improvement directions.

---

## 9. Reproducibility

To recreate dataset:

```python
from job_intel.pipelines.chapter0_build_base_dataset import build_chapter0_base_dataset
df = build_chapter0_base_dataset()
```

Output:

```
data/processed/ch0_processed_jobs.csv
```

---

## 10. Final Notes

Chapter 0 successfully transforms raw Glassdoor text into a **fully engineered, analysis-grade dataset** with transparent provenance and reproducible transformations.

This file serves as the **official data dictionary** for the Job Intelligence Engine.

