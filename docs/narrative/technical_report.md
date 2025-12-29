# Job Intelligence Engine — Technical Report
Version: YYYY-MM-DD  
Repo: <link>  
Live App: <link>  
Author: Alejandro de la Fuente  
License: <MIT/Apache-2.0/etc>  
Data access: <public / not redistributable / instructions>

---

## 0. Abstract (150–250 words)
Purpose: pseudo-academic summary that stands alone.
- Problem, approach, data, main outputs, headline results, and key limitations.

---

## 1. Executive Summary (1 page)
Purpose: hiring-manager-readable summary.
- What the system does (in one paragraph).
- What it produces (bullets).
- Why it’s credible (evidence bullets).
- Key results snapshot (metrics + behavior-based checks).
- Where to see it: app link + demo scenario.

**Figure 1:** System overview diagram (engine + artefacts + app).

---

## 2. Introduction
Purpose: frame the market problem and why it’s hard.
- Motivation: noisy job titles, inconsistent skills, salary variation, candidate uncertainty.
- Prior/typical approaches (brief) and what this system adds.
- Research questions / engineering goals (3–6 bullets).
- Scope boundaries and non-goals (e.g., no forecasting, no causal claims).

---

## 3. System Overview and Contributions
Purpose: what’s novel and what you built end-to-end.
- Core contributions (pipeline, models, embeddings, positioning, recommender, macro overlay).
- Design principles (reproducibility, explicit contracts, deterministic artefacts).
- High-level artefact map (what is saved and reused).

**Figure 2:** Artefact flow / dependency graph (chapters as modules, not silos).

---

## 4. Data and Provenance
Purpose: show realism about data.
- Sources, collection method, coverage.
- Inclusion/exclusion rules.
- Salary parsing / normalization, deduplication, missingness policy.
- Train/validation/test strategy and leakage controls.

**Table 1:** Dataset overview.  
**Table 2:** Missingness + key field quality checks.

---

## 5. Unified Methods (End-to-End Narrative)
Purpose: “Methods” written as one coherent system (not chapter fragments).
Describe the full transformation chain with inputs → outputs → rationale:
1) Title normalization  
2) Skill extraction + skill families  
3) Salary modelling  
4) Skill requirement modelling  
5) Market geometry: job–skill graph → embeddings → job families  
6) Individual positioning: suitability, competitiveness, gap analysis  
7) Recommender: best_now/stretch + salary alignment  
8) Counterfactual upskilling  
9) Macro overlay: adjacent families + co-learning skills

**Figure 3:** One-page “engine blueprint” (boxes + arrows + artefact names).

---

## 6. Representation Layer
Purpose: document the “language” of the engine.
### 6.1 Job Title Normalization
- Taxonomy definition (role + seniority).
- Normalization method, edge cases, error taxonomy.

### 6.2 Skill Extraction and Skill Families
- Tokenization/parsing approach.
- Skill family ontology and how it’s used (multi-hot + probability).
- Coverage and expected false negatives.

**Table 3:** Skill family summary (count, examples).  
**Appendix link:** full skill dictionary.

---

## 7. Predictive Modelling Layer
Purpose: document modelling choices and evaluation.
### 7.1 Salary Model
- Objective and target definition.
- Feature sets and encoding.
- Training procedure, metrics, calibration/stability notes.

### 7.2 Skill Requirement Models
- Objective: per-skill probability.
- Training/eval approach, thresholding policy (if any).
- How these probabilities feed downstream components.

**Table 4:** Salary model performance (overall + slices).  
**Table 5:** Skill model performance (overall + slices + hardest skills).

---

## 8. Market Structure Layer (Hidden Geometry)
Purpose: justify latent structure artefacts.
### 8.1 Job–Skill Bipartite Graph
- Nodes/edges/weights; why weighted.

### 8.2 Embeddings and Job Families
- Embedding method + clustering method.
- Model selection (K choice) + stability checks.

### 8.3 Skill Co-occurrence Network
- Co-occurrence definition, weighting, filtering.
- Intended use (co-learning neighbours, bundle discovery).

**Figure 4:** Embedding space visualization (optional).  
**Figure 5:** Example co-occurrence subgraph (optional).

---

## 9. Individual Positioning (User → Market)
Purpose: turn market artefacts into user-conditioned signals.
- User profile schema + validation.
- Candidate retrieval/filters.
- Suitability definition and interpretability.
- Competitiveness definition and rationale.
- Skill gap analysis and severity ranking.

**Table 6:** Positioning output schema.  
**Figure 6:** Suitability vs competitiveness example (optional).

---

## 10. Recommendation and Upskilling Engine
Purpose: decision support logic.
### 10.1 Recommendations (best_now vs stretch)
- Definitions, guardrails, composite scoring and tie-breaking.
- How salary alignment is used (and how not to overtrust it).

### 10.2 Counterfactual Upskilling
- Counterfactual loop specification (add one skill → recompute deltas).
- Ranking objective (suitability gain, competitiveness reduction, salary-gap closing).
- Reporting layer: skill families + demanded tokens in target jobs.

**Table 7:** Recommendation output schema.  
**Figure 7:** Example upskilling delta plot (optional).

---

## 11. Macro Lens Overlay (Context, Not Re-ranking)
Purpose: broaden options while keeping the core engine stable.
### 11.1 Adjacent Job Families
- Mapping jobs → family.
- Nearest family retrieval (centroid distance).
- Presentation (credible alternatives, not new ranking logic).

### 11.2 Co-learning Neighbours
- Top co-occurring skills for each recommended upskill.
- Within-family vs adjacent-family variants.

**Table 8:** Nearest families + representative titles.  
**Figure 8:** Family neighbourhood view (optional).

---

## 12. Evaluation, Baselines, and Ablations
Purpose: the strongest credibility section.
### 12.1 Evaluation Framework
- What “passes” means for each layer (models, clustering, positioning, recommender).
- Determinism and reproducibility checks.

### 12.2 Baselines
- Naive recommender baseline (e.g., salary-only, suitability-only, popularity proxy).
- Baseline positioning (no competitiveness / no rarity).
- Baseline upskilling (frequency-only).

### 12.3 Ablations (Module Value)
- Remove salary signal: what changes.
- Remove competitiveness: what changes.
- Remove job families / macro overlay: what changes.
- Remove counterfactual recompute: what changes.

### 12.4 Slice and Stress Testing
- By state/city, seniority, industry, job family.
- Sparse profile behavior and failure modes.

**Table 9:** Baseline comparison results.  
**Table 10:** Ablation deltas (impact summary).

---

## 13. Results, Discussion, and Conclusions (Unified)
Purpose: interpret findings and show mature judgment.
- What worked well (evidence-backed).
- Where the engine is most reliable and why.
- Key trade-offs (e.g., precision vs coverage; salary noise vs usefulness).
- Known failure modes and how they present in outputs.
- Implications for users (how to act on recommendations responsibly).
- Conclusions: what this project demonstrates technically.

**Figure 9:** End-to-end demo scenario outputs (1–2 pages).  

---

## 14. Deployment and Public App
Purpose: show productization.
- Platform choice and rationale.
- UX flow (inputs → run → outputs tabs).
- Caching/performance, artefact loading strategy.
- Privacy stance (what is stored, what is not).
- Hosting and minimal CI/CD (even lightweight).

**Figure 10:** App screenshots (3 panels).  
**Appendix link:** app run/deploy instructions.

---

## 15. Engineering Quality: Reproducibility, Testing, and Contracts
Purpose: highlight professional engineering standards explicitly.
- Pipeline orchestration and artefact registry.
- Invariant suite and schema contracts.
- Error handling philosophy and user-facing validation.
- Monitoring-style checks (if any).

**Table 11:** QA checklist with evidence links.

---

## 16. Assumptions, Limitations, and Ethical Use
Purpose: mature stance (not apologetic).
- Data bias and representativeness.
- Salary measurement limitations (total comp, benefits, reporting bias).
- Skill extraction limits (coverage, false negatives).
- Interpretability limits (association ≠ causation).
- Appropriate use statement and misuse risks.

---

## 17. Future Work (Prioritized Roadmap)
Purpose: prove you can product-manage scope.
- High-ROI next (2–5 items).
- Nice-to-have (optional).
- What additional data/infrastructure would enable (explicit dependencies).

---

## 18. How to Reproduce (Quickstart)
Purpose: a stranger can run it.
- Environment setup.
- One-command runs (per pipeline).
- Expected artefacts and where to find them.
- Seeds, determinism notes.
- How to regenerate figures/tables.

---

## References
Purpose: cite key methods (node2vec, SHAP, etc.) and any external datasets.

---

# Appendices

## Appendix A — Model Cards (Standardized)
Purpose: each core component has a “model card” (even if not ML-heavy).

### A1. Model Card — Salary Model
- Intended use / users
- Training data summary
- Features
- Target definition
- Training procedure
- Metrics (overall + slices)
- Calibration / stability notes
- Known failure modes
- Ethical considerations

### A2. Model Card — Skill Requirement Models
(same headings; include macro-averages + per-skill extremes)

### A3. Model Card — Embeddings + Job Family Clustering
- What it represents / doesn’t represent
- Training procedure
- Validity checks (stability, interpretability)
- Limitations and drift risks

### A4. Model Card — Suitability / Competitiveness Indices
- Definitions, intended use, sensitivity notes
- Failure modes for sparse/noisy profiles

### A5. Model Card — Recommender (Composite Scoring)
- Inputs/outputs
- Scoring components + guardrails
- Known failure modes

### A6. Model Card — Upskilling Counterfactual Engine
- Objective and ranking criteria
- Where it is reliable (and where not)
- Failure modes

### A7. Model Card — Macro Lens Overlay
- Neighbour families logic
- Co-learning logic
- Misinterpretation risks

---

## Appendix B — Data Dictionary
(link or embedded; reference-only)

## Appendix C — User Profile Schema
(contract for app/API)

## Appendix D — Artefact Registry
- Artefact name → description → filepath → produced by pipeline step

## Appendix E — Figures and Tables Index
- Figure/Table number → description → filepath

## Appendix F — Glossary
- Suitability, competitiveness, job family, skill token, etc.

## Appendix G — Changelog
- Major milestones/releases and what changed.
