# Architecture — Chapter 0: Data Ingestion & Feature Foundations

This document describes the architecture, data flow, and design rationale for **Chapter 0** of the Job Intelligence Engine.
Chapter 0 is the foundation of the entire project: it ingests raw job postings, standardises all text inputs, extracts structured features, and produces the **canonical dataset** used downstream in Chapters 1–5.

---

## 1. Purpose of Chapter 0

Chapter 0 solves three fundamental problems:

1. **Raw job postings are messy, inconsistent, and incomplete.**  
   We unify them into a single, clean, structured dataset.

2. **Key information needed for modelling (skills, seniority, title, salary) is not available as structured fields.**  
   We engineer these signals explicitly.

3. **All downstream modelling must rely on a stable, consistent dataset.**  
   Chapter 0 defines that dataset.

---

## 2. High-Level Architecture

```
Raw Data (DA + DS CSVs)
        │
        ▼
Load & Harmonise Inputs
        │
        ▼
Minor Feature Cleaning
        │
        ▼
Description Cleaning
        │
        ▼
Title Features (seniority, family)
        │
        ▼
Domain Lookup
        │
        ▼
Salary Parsing
        │
        ▼
Skill Extraction
        │
        ▼
NA Handling & Pruning
        │
        ▼
Final ch0_processed_jobs.csv
```

---

## 3. Module-Level Architecture

### 3.1 `load_raw_jobs()`
Loads the Data Analyst and Data Scientist CSVs, aligns their columns, standardises NAs, and tags each row with its original source.

### 3.2 `minor_feature_cleaning()`
Adds:
- `state`
- `state_hq`
- `ownership_clean`
- Cleaned `Founded`
- Drops: Easy Apply, Competitors, Company Name, Revenue, Type of ownership, Location, Headquarters

### 3.3 `add_description_features()`
Produces:
- `job_description_clean`

Includes regex cleaning, punctuation removal, and stopword removal to ensure consistency for NLP-style extraction.

### 3.4 `add_title_features()`
Produces:
- `job_title_base`
- `seniority_roman`
- `seniority_title`
- `seniority_description`
- `seniority_combined`
- `job_title_norm`
- `job_title_family`
- `domain`

### 3.5 `add_salary_features()`
Parses salary ranges and standardises hourly → annual values.  
Outputs:
- `sal_min`
- `sal_max`
- `sal_mean`

### 3.6 `add_skill_features()`
Outputs ~20 domain–level skill indicators using the v1 taxonomy.

### 3.7 Final NA Handling & Column Pruning
- Drop rows with **no salary AND no description**
- Fill missing Industry / Sector / Size with “Unknown”
- Drop raw and intermediate columns

---

## 4. Final Output Dataset: Schema Summary

- **47 columns**
- **6162 rows**
- No structural NA other than:
  - Rating (left for modelling)
  - Founded (left for modelling)

---

## 5. Status

Chapter 0 is **complete, stable, and production-ready**.

