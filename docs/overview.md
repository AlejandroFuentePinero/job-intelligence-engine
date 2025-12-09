# Job Intelligence Engine — Project Overview  
Verified narrative for Chapter 0 + Salary Model (Chapter 1).  
Date: 2025-12-09

The Job Intelligence Engine transforms raw job postings into structured, model-ready data and builds predictive systems (starting with salary modelling).  
This document explains the conceptual rationale behind Chapter 0 and the first component of Chapter 1.

---

# Chapter 0 — Narrative & Purpose

Job postings are messy: inconsistent titles, ambiguous salaries, noisy text, and unstructured skill references.  
Chapter 0 standardises this chaos into a **clean, unified, modelling-ready dataset**.

Its design principles:

### 1. Unify heterogeneous sources  
Data Analyst and Data Scientist postings are merged into a consistent schema.

### 2. Standardise role identity  
Extract meaningful abstractions: cleaned title, job family, seniority.

### 3. Extract semantic signal  
Clean descriptions, assign domain labels via SBERT lookup, and derive multi-hot skill attributes from a curated taxonomy.

### 4. Normalise salary data  
Convert raw, inconsistent salary strings into annualised numeric ranges.

### 5. Build a complete and reproducible feature space  
Chapter 0 output forms the backbone of all downstream modelling.

---

# Chapter 1 — Salary Response Model (Narrative)

The first system mechanic is predicting salary from job attributes.

Guiding principles:

### 1. Skills are sparse and correlated  
Binary flags alone do not capture structure → PCA extracts latent dimensions (e.g., analytics vs engineering orientation, ML depth).

### 2. Categorical attributes encode systematic salary variation  
Company size, sector, ownership, state, seniority, and enriched title carry predictable signals.

### 3. Prefer robust, interpretable ML  
XGBoost is used for its strength on mixed feature types and non-linear interactions.

### 4. Ready for future integration  
The model is wrapped in a clean prediction interface for downstream use:
- Suitability scoring  
- Fairness and bias analysis  
- Job embeddings  
- Recommendations and optimisation  

---

# How Chapters Fit Together  
- **Chapter 0** provides the standardised representation of every job.  
- **Chapter 1** begins modelling that world by quantifying salary.  

Later chapters will build embeddings, competitiveness scores, skill value modelling, job–skill graphs, and career-path optimisation.

This overview captures the conceptual story behind what has been built so far.
