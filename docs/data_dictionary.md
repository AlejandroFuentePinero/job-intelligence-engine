# Job Intelligence Engine — Data Dictionary  
Verified against jobs_ch0.csv  
Date: 2025-12-09

---

# 1. Dataset Summary  
**Source:** Raw Glassdoor datasets (Data Analyst + Data Scientist)  
**Final Output:** `ch0_processed_jobs.csv`  
**Rows:** 6,162  
**Columns:** 47  

---

# 2. Column Definitions (Chapter 0)

## A. Company Metadata

| Column | Type | Definition |
|--------|------|------------|
| `Rating` | float (nullable) | Glassdoor rating (0–5). |
| `Size` | str | Company size category (`Unknown` if missing). |
| `Founded` | Int64 | Nullable year founded. |
| `Industry` | str | Company industry (`Unknown`). |
| `Sector` | str | High-level sector (`Unknown`). |
| `ownership_clean` | str | Standardised ownership. |
| `state` | str | US state or `"international"`. |

---

## B. Role Identity

| Column | Type | Definition |
|--------|------|------------|
| `seniority_combined` | str | Seniority extracted from title + description. |
| `job_title_family` | str | Normalised job family. |
| `domain` | str | Domain label from SBERT lookup. |
| `role_source` | str | DA or DS dataset origin. |

---

## C. Salary Features

| Column | Type | Definition |
|--------|------|------------|
| `sal_min` | float | Annualised salary minimum. |
| `sal_max` | float | Annualised salary maximum. |
| `sal_mean` | float | Salary midpoint (target). |
| `sal_is_hourly` | bool | Whether original salary was hourly before annualisation. |

---

## D. Skill Flags (0/1)

### Core Programming  
- `core_programming__basic`  
- `core_programming__intermediate`  
- `core_programming__advanced`  

### Data Engineering & Pipelines  
- `data_engineering_pipelines__basic`  
- `data_engineering_pipelines__intermediate`  
- `data_engineering_pipelines__advanced`  

### Machine Learning & AI  
- `ml_ai__basic`  
- `ml_ai__intermediate`  
- `ml_ai__advanced`  

### Analytics & Statistics  
- `analytics_stats__basic`  
- `analytics_stats__intermediate`  
- `analytics_stats__advanced`  

### BI & Visualisation  
- `bi_viz__basic`  
- `bi_viz__intermediate`  
- `bi_viz__advanced`  

### Cloud / MLOps  
- `cloud__basic`  
- `cloud__intermediate`  
- `cloud__advanced`  

### Databases & Storage  
- `db_storage__basic`  
- `db_storage__intermediate`  
- `db_storage__advanced`  

### Productivity & Workflow  
- `productivity_workflow__basic`  
- `productivity_workflow__intermediate`  
- `productivity_workflow__advanced`  

### Soft Skills  
- `soft_skills__core`  
- `soft_skills__leadership`  

### Domain-Specific  
- `domain_specific__none`  

---

## E. Text Fields

| Column | Meaning |
|--------|---------|
| `Job Description` | Raw job description. |
| `job_description_clean` | Cleaned description for extraction. |
| `job_title_base` | Cleaned title without noise or seniority. |
| `job_title_norm` | Normalised version for lookup. |
| `title_plus_description` | Combined text used for skills. |

---

# 3. Missing Value Policy

- Drop rows missing **both** salary and description  
- Fill with `"Unknown"`: Size, Industry, Sector  
- Keep NA: Rating, Founded  
- Skill flags: always 0/1  
- Salary fields: allow NA  

---

# 4. Salary model (Chapter 1) Features (Engineered)

## Categorical Encodings  
- `size_code`  
- `sector_code`  
- `state_code`  
- `ownership_code`  
- `seniority_code`  
- `title_rich_code`

## PCA Components  
- `skill_PC1` … `skill_PC10`

## Optional One-Hot Dummies  
Generated during experimentation only.

