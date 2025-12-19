# Job Intelligence Engine — Progress Tracker (STEERING DOCUMENT)
_Last updated: 2025-12-15_

> This document is the **authoritative steering and sanity-check document**.
> It reflects what is required, what is optional, and what must be translated
> from notebooks into production `src/` code.

---

# OVERALL PROJECT PROGRESS
[#########.............] 45%
~45% complete

---

# ✅ Chapter 0 — Foundations (Preprocessing & Taxonomy)

**Progress:** 100%  
`[████████████████████] 100%`

### Core Work (LOCKED)
- [x] Clean raw data  
- [x] Title normalisation  
- [x] Seniority extraction  
- [x] Domain assignment  
- [x] Skill extraction  
- [x] Salary parsing  
- [x] Final processed dataset  
- [x] Documentation complete  

### Pipelines / Architecture
- [x] **Chapter 0 Processing Pipeline** (raw → engineered data)
- [x] `titles.py`
- [x] `seniority.py`
- [x] `skills.py`
- [x] `salary_parser.py`
- [x] `taxonomy_builder.py`

---

# ✅ Chapter 1 — System Mechanics

**Progress:** 100%  
`[████████████████████] 100%`

> Chapter 1 produces **mechanical, reusable signals** consumed downstream.
> No user-level optimisation or decision logic occurs here.

---

## 1.1 Salary Response Model

### Core Work
- [x] Feature engineering  
- [x] Skill dimensionality reduction via PCA (10 components)  
- [x] Train salary model (XGBoost v4)  
- [x] Evaluation & diagnostics  
- [x] Documentation added  

### Pipelines / Architecture
- [x] `pca_transformer.py`
- [x] `salary_model.py`
- [x] `salary_evaluator.py`
- [x] **Salary Modelling Pipeline**  
  (features → PCA → prediction → residuals)

---

## 1.2 Skill Requirement Models

### Core Work
- [x] Train 27 skill models  
- [x] Evaluation metrics  
- [x] Skill probability matrix  
- [x] Save outputs  
- [x] Documentation added  

### Pipelines / Architecture
- [x] `skill_models.py`
- [x] `skill_matrix_builder.py`
- [x] **Skill Requirement Pipeline**

---

## 1.3 Salary Fairness Analysis

### Core Work
- [x] Residuals computed  
- [x] Group summaries (state, sector, size, ownership, seniority)  
- [x] Diagnostic plots  
- [x] Interpretation written  

### Pipelines / Architecture
- **No pipeline required**  
  (one-off diagnostic aggregation from salary model residuals)

---

## 1.4 Explainability Suite

### Core Work
- [x] SHAP global importance  
- [x] SHAP dependence plots  
- [x] PDP  
- [x] ICE  
- [x] Integrated into report  

### Pipelines / Architecture
- **No pipeline required**  
  (explainability derived directly from trained salary model)

---

## 1.5 Skill Value

### Core Work
- [x] Derive Global Skill Value Index  
      (SHAP × PCA back-projection → skill-level signal)
- [x] Save `skill_value_index.csv`
- [x] Documentation paragraph (short, defensive)

### Pipelines / Architecture
- **No pipeline required**  
  (static interpretive artefact; not consumed downstream)

---

## 1.6 Chapter 1 Consolidation

### Core Work
- [x] Documentation consistency pass  
- [x] Pipeline validation checks  
- [x] Define **Chapter 1 → Chapter 2 data contract**
- [x] List guaranteed saved artefacts + schemas  
- [x] Update `requirements.txt`  
- [x] Close Chapter 1  

### Pipelines / Architecture
- [x] Pipeline to run both model pipelines 

---

# ✅ Chapter 2 — Hidden Structure

**Progress:** 100%  
`[████████████████████] 100%`

### Core Work
- [x] Build job–skill bipartite graph  
- [x] Train Node2Vec embeddings  
- [x] Job family clustering  
- [x] Skill co-occurrence network
- [x] Industry Specialization Maps (clustered skill vectors)
- [x] Documentation


### Pipelines / Architecture
- [x] `graph_builder.py`
- [x] `node2vec_trainer.py`
- [x] `embedding_loader.py`
- [x] `job_clusterer.py`
- [x] `skill_embedding_similarity.py`
- [x] **Embedding Pipeline**  
  (graph → embeddings → clustering)
- [x] Add regression tests (shapes, determinism)

---
# ✅ CHAPTER 3 — Individual Positioning

**Progress:** ~100%  
`[████████████████████] 100%`

---

## Core Work
- [x] **User profile schema**  
- [x] **Candidate set construction**  
- [x] **Suitability components (definition + normalization)** 
- [x] **Job suitability score**  
- [x] **Competitiveness index**  
- [x] **Skill gap analysis**  
- [x] **Skill rarity integration (inverse frequency)**  
- [x] **Competitiveness sensitivity analysis**  
- [x] **Suitability sensitivity analysis**  
- [x] **Public API**  
- [x] **Chapter 3 documentation**  

---

## Pipelines / Architecture
- [x] `schemas.py`  
- [x] `user_profile_skill_processor.py`  
- [x] `artefacts.py`  
- [x] `candidate_selection.py`  
- [x] `candidate_suitability.py`  
- [x] `skill_rarity.py`  
- [x] `candidate_competitiveness.py`  
- [x] `candidate_skill_gap.py`  
- [x] `competitiveness_sensitivity.py`  
- [x] `suitability_sensitivity.py`  
- [x] `positioning.py`  
- [x] Pipeline entrypoint `pipelines/ch3_individual_positioning.py`  
- [x] **Deterministic tests**  
  *(golden fixtures, invariance checks, empty-set behavior)*

---

# Chapter 4 — Recommender Engine

**Progress:** 0%  
`[....................] 0%`

### Core Work
- [ ] Hybrid job recommender  
- [ ] Upskilling recommender  
- [ ] Salary prediction based on user?

### Pipelines / Architecture
- [ ] `job_recommender.py`
- [ ] `upskilling_recommender.py`
- [ ] `roi_estimator.py`
- [ ] **Recommendation Pipeline**
- [ ] End-to-end recommender tests

---

# Chapter 5 — Insights & Dashboards (OPTIONAL)

**Progress:** 0%  
`[....................] 0%`

### Core Work
- [ ] Salary landscape dashboard  
- [ ] Skill ecosystem visualisation  
- [ ] Geographic summaries  

### Pipelines / Architecture
- **No pipeline required**  
  (visualisation layer only)

---

# LOCKED PROJECT GUARANTEES

- Every module explicitly states pipeline status  
- Notebook → `src/` translation is tracked where required  
- Optional work cannot silently become required  
- This document is the single source of truth for scope and progress
