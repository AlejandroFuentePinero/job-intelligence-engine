# Job Intelligence Engine — Progress Tracker
_Last updated: 2026-01-01_

> This document is the **authoritative steering and sanity-check document**.
> It reflects what is required, what is optional, and what must be translated
> from notebooks into production `src/` code.

---

# OVERALL PROJECT PROGRESS (V1)
[███████████████████.] 99%
~99% complete

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
# ✅ Chapter 4 — Recommender Engine

**Progress:** 100% (implemented; validation pending)  
`[████████████████████] 100%`

## Core Work
- [x] Define recommendation intents + scoring objectives (v1)
- [x] Candidate universe + filtering contract (canonical eligible jobs table)
- [x] Chapter 4 context loader (wrapper) + salary feature construction (PC broadcast)
- [x] Chapter 4 entrypoint evaluation (salary features + predict smoke + alignment)
- [x] Hybrid job recommender (retrieve → rerank; 2 buckets)
- [x] Explanation layer (why job / why skill; per-bucket rationale)
- [x] Upskilling recommender (counterfactual skill addition → positioning gain on frozen universe)
- [x] ROI proxy (v1): `upskill_impact_score` (promotion-rate + score-gain composite with demotion/tail penalties + guardrail)
- [x] Career simulation module (scenario-based what-if; implemented)
- [x] Persist Chapter 4 demo (config + optional golden outputs)
- [x] Documentation (module docs + assumptions + limitations)

## Pipelines / Architecture
- [x] `features/artefacts_ch4.py`
- [x] `evaluation/chapter_4_entrypoint_eval.py`
- [x] `features/job_recommender.py`
- [x] `features/job_explanations.py`
- [x] `features/upskilling_recommender.py`
- [x] `v2_updates/features/career_simulator.py`
- [x] `pipelines/chapter4_recommender.py` *(canonical orchestrator)*

## Validation & Testing (gate to 100%)
- [x] `evaluation/ch4_invariants.py` (frozen-universe checks, shape/keys contracts, failure-mode guards)
- [x] End-to-end recommender tests (demo JSON run + smoke assertions; optional golden snapshot comparison)

---

# ✅ Chapter 5 — Insights & Dashboards

**Progress:** ~100%  
`[████████████████████] 100%`

## Core Work

### Product Surface (App Spine)
- [x] App shell + navigation (4–6 pages max; demo-first)
- [x] Demo personas loader (1–2 preset user profiles + reset)
- [x] One-click “Load demo + take me to Recommender” path (no configuration required)
- [x] Input validation + guardrails (empty results, no candidates under constraints, bad inputs)

### Market Mechanics (Chapter 1 outputs)
- [x] Salary landscape dashboard *(Landscape page in place)*
- [x] Model performance snapshot (R²/RMSE/MAE + residual histogram) *(residual hist yes; metrics snapshot not wired)*
- [x] Fairness residual summaries
- [x] SHAP global importance
- [x] Global skill value ranking

### Market Structure (Chapter 2 outputs)
- [x] Skill ecosystem visualisation (co-occurrence) *(via upskilling macro co-learning plot)*
- [x] Skill neighbour lookup (top-k neighbours per skill; simple + fast) *(wired from similarity edges for top upskill skills)*

### User & Recommendation Context (Chapter 3–4 outputs)
- [x] User positioning summary *(not a dedicated panel; implicit in recommender inputs/outputs)*
- [x] Recommendation views (`best_now`, `stretch`)
- [x] Job-level “why” panel (covered vs missing families; key drivers; salary-gap context)
- [x] Upskilling plan views (counterfactual deltas; top 3)

### Macro Recommendation Layer (Chapter 2 → optional, shown in Chapter 5)
- [x] Macro vs micro framing (micro = constraint-defined market; macro = global exploration; opt-in) *(macro section added in upskilling page)*
- [x] Co-learning neighbours for upskilling (top 3 skills; 2–3 neighbours each; hard caps) *(implemented as top-3 focals × top-5 neighbours plot)*

## Pipelines / Architecture
- [x] `pipelines/ch5_app_build.py` (single entrypoint to assemble app-ready artefacts)
- [x] Artefact manifest (required outputs from Ch1/Ch2/Ch4 + load locations)
- [x] Smoke tests (run build + open app + load demo personas)
- [x] Chapter 5 reproducibility notes (data sources + joins + caps + assumptions)

---

# Chapter 6 — V1 Final Steps (Shipping + Portfolio)
**Progress:** ~80%  
[███████████████.....] 80%

## Core deliverables (LOCKED)
- [x] Portfolio write-up (unified narrative)
- [x] Project README consolidation (single front door)
- [x] How to run (one command) + environment notes
- [x] requirements.txt review + pinning strategy (prod vs dev if needed)
- [x] gitignore review
- [x] Dictionary documents
- [x] Architecture documents
- [x] App short GIF (demo path)
- [x] Explicit v2 improvement list (scoped, ranked)
- [x] Licensing / attribution (dataset sources; external assets)

## Shipping proof points (HIGH SIGNAL)
- [x] CI smoke check (GitHub Actions): run `python -m src.job_intel.evaluation.ch5_smoke_test`
- [x] App “build info” stamp (git commit hash + artefact date/version shown in UI) *(implemented: `get_build_info()` + sidebar panel in `app.py`)*

## Optional (only if fast)
- [ ] Deploy app online (Streamlit Cloud / HF Spaces / Render) OR publish local-only with demo video
- [ ] LinkedIn update (project + 2-paragraph description)
- [x] Portfolio entry
- [ ] Final scope lock (explicit “v1 is closed” statement in README)

---

# OVERALL PROJECT PROGRESS (V2)
[█....................] 2%
~2% complete

---

# V2 review
- [ ] V2 integration plan

---

# Chapter 4
- [x] Career simulation engine (Chapter 4)
