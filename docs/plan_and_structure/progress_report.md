# Job Intelligence Engine — Progress Tracker  
_Last updated: 2025-12-13_

---

# OVERALL PROJECT PROGRESS
`[######................] 30%`  
**~30% complete**

---

# Chapter 0 — Foundations (Preprocessing & Taxonomy)

**Progress:** 100%  
`[####################] 100%`

### Core Work
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
- [x] Title module  
- [x] Skill extractor  
- [x] Salary parser  
- [x] Domain lookup
- [x] Seniority parser
- [x] Text cleaning helper
- [x] Skill taxonomy builder
- [x] Processing pipeline benchmark tester

---

# Chapter 1 — System Mechanics  
**Progress:** ~90%  
`[##################..] 90%`

---

## 1.1 Salary Modelling

### Core Work
- [x] Feature engineering  
- [x] PCA (10 components)  
- [x] Train salary model (XGBoost v4)  
- [x] Evaluation + diagnostics  
- [x] Documentation added  

### Pipelines / Architecture
- [x] PCA transformer  
- [x] Salary predictor  
- [x] Salary model evaluator  
- [x] **Salary Modelling Pipeline**  
  (features → PCA → model → predictions → residuals)

---

## 1.2 Skill Requirement Models

### Core Work
- [x] Train 27 LightGBM models  
- [x] Evaluation metrics  
- [x] Skill probability matrix  
- [x] Save evaluation + matrix  
- [x] Documented  

### Pipelines / Architecture
- [x] Skill model evaluator  
- [x] Probability matrix builder  
- [x] **Skill Requirement Pipeline**  
  (binary flags → 27 models → probability matrix)

---

## 1.3 Salary Fairness Analysis

### Core Work
- [x] Residuals computed  
- [x] Fairness tables (state, sector, size, ownership, seniority, family)  
- [x] Weighted + unweighted plots  
- [x] Interpretations written  
- [x] Documentation updated  

### Pipelines / Architecture
- No pipeline required (intentionally EDA-only)

---

## 1.4 Explainability Suite — CURRENT

### Core Work
- [x] SHAP global  
- [x] SHAP dependence plots  
- [x] PDP plots  
- [x] ICE curves  
- [x] Add to project report and overview 

### Pipelines / Architecture
- No pipeline required (intentionally EDA-only)

---

## 1.5 Skill Value Ranking — UPCOMING

### Core Work
- [ ] Define ranking logic (SHAP × fairness × skills)  
- [ ] Produce sector/title/state tables  
- [ ] Documentation added  

### Pipelines / Architecture
- [ ] Skill value ranking utility  
- [ ] **Skill Value Ranking Pipeline**  
  (SHAP + fairness + PCA → ranking tables)

---

## 1.6 Chapter 1 Consolidation  
- [ ] Documentation consistency pass  
- [ ] Finalise all saved artefacts  
- [x] Pipeline and model validations
- [ ] Define Chapter 1 → Chapter 2 data contract  
      (list & schema of saved outputs used downstream)  
- [x] Simple Chapter 1 regression test  
      (run salary + skill pipelines, check metrics/shapes)  
- [ ] Ensure env/requirements updated (xgboost, lightgbm, shap)  
- [ ] (Optional) Chapter 1 Unified Pipeline wrapper  
- [ ] Update progress tracker  
- [ ] Close chapter


---

# Chapter 2 — Hidden Structure (Embeddings & Graphs)

**Progress:** 0%  
`[....................] 0%`

### Core Work  
- [ ] Job–skill bipartite graph  
- [ ] Node2Vec embeddings  
- [ ] Job clustering  
- [ ] Skill co-occurrence network  

### Pipelines / Architecture
- [ ] Graph builder  
- [ ] Node2Vec trainer  
- [ ] Embedding loader  
- [ ] Clustering utilities  
- [ ] **Embedding Pipeline**  
  (graph → Node2Vec → embeddings → clustering)

---

# Chapter 3 — Individual Positioning

**Progress:** 0%  
`[....................] 0%`

### Core Work
- [ ] Job Suitability Score  
- [ ] Competitiveness Index  
- [ ] Skill Gap Analysis  

### Pipelines / Architecture
- [ ] Suitability scorer  
- [ ] Competitiveness calculator  
- [ ] Gap analysis tool  
- [ ] **Suitability Pipeline**  
  (user skills → embeddings → suitability)  
- [ ] **Competitiveness Pipeline**  
  (salary/fairness/skill rarity → difficulty metric)

---

# Chapter 4 — Recommender Engine

**Progress:** 0%  
`[....................] 0%`

### Core Work
- [ ] Hybrid job recommender  
- [ ] Upskilling recommender  

### Pipelines / Architecture
- [ ] Recommender engine  
- [ ] ROI calculator  
- [ ] Simulation util
