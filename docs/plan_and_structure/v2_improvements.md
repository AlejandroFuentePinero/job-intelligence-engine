# V2 Improvements Backlog

This backlog captures **plausible v2 extensions** beyond the v1 deliverable.  
v1 is intentionally constrained to a **deterministic, auditable, end-to-end system** with strong contracts and evaluation. Most v2 items are deferred because they (i) require external data, (ii) materially expand surface area + testing burden, or (iii) add UX complexity without proportional portfolio value at v1.

**V2 inclusion rule:** ship an item only if it either (a) measurably improves recommendation quality, (b) materially improves user understanding, or (c) enables a new credible decision-support use case.

---

## Chapter 0 — Foundations (Preprocessing & Taxonomy)

### Candidate improvements
- **Title embeddings + clustering refresh** (retrain or re-cluster title/domain lookup if coverage drifts).
- **Skill dictionary expansion + normalisation** (synonyms, typos, canonical token mapping).
- **Salary extraction hardening** (more robust parsing; better recovery when salary appears in description).
- **Experience requirement extraction** (years-of-experience, degree, “required vs preferred” patterns).
- **NER / transformer-based skill extraction** (reduce reliance on pure dictionary matching; improve recall/precision).
- **Domain-specific fine-tuning (SBERT)** for titles/skill tokens *only if* clustering quality becomes a bottleneck.
- **Stronger seniority detection** (messy titles; consistent mapping across job families).
- **International expansion entrypoint** (salary normalisation + currency translation + geo handling).

### Why deferred in v1
These changes alter the base feature layer and therefore shift all downstream outputs (Ch1–Ch4). They are high leverage but high risk: any upgrade requires re-running benchmarks, re-validating contracts, and re-evaluating recommendation behaviour end-to-end.

---

## Chapter 1 — System Mechanics (Salary, Skill Demand, Fairness)

### Candidate improvements
- **Salary model upgrades** (new features, alternative objectives, alternative model families).
- **Uncertainty estimates** for salary and skill-demand models (calibration + intervals, not just point estimates).
- **Quantile regression / prediction intervals** for salary (P10/P50/P90 instead of only mean prediction).
- **Stronger generalisation validation** (grouped splits by company/title cluster; stress-test leakage risks).
- **Segmented skill-value** views (city/sector/title-specific marginal value rather than global-only).

### Why deferred in v1
These are “model quality” upgrades that can spiral into open-ended iteration. In v1, a point-estimate salary signal is sufficient as an **interpretable alignment cue** in the recommender; deeper model improvements are better treated as a focused refinement pass after the full app surface is shipped and stable.

---

## Chapter 2 — Hidden Structure (Graphs, Clusters, Ecosystems)

### Candidate improvements
- **Alternative clustering for job families** (e.g., HDBSCAN) if KMeans families remain hard to interpret.
- **Family/sector specialisation profiles** (clustered skill vectors; compact “what defines this family” cards).
- **Alternative embeddings** (e.g., contrastive/Siamese SBERT) if Node2Vec is too graph-structure-driven.

### Why deferred in v1
These are macro-market enhancements that mainly improve interpretability and exploration. v1 only needs stable **skill similarity** for user-conditioned co-learning; richer job-family storytelling becomes v2 once you can label families cleanly and prove the extra complexity helps users.

---

## Chapter 3 — Individual Positioning (Suitability, Competitiveness, Gaps)

### Candidate improvements
- **Activate rarity weighting in scoring** (v1 computes rarity artefact; v2 can integrate it into competitiveness and/or gap severity with documented impact).
- **Skill difficulty / cost signal** (O*NET or proxy difficulty) for “gap cost”, not only “gap size”.
- **User sensitivity diagnostics** (how rankings change when adding/removing specific skills; scenario-by-skill deltas).
- **Location summaries** (city/state competitiveness rollups; concentration summaries for constraint-aware views).

### Why deferred in v1
Difficulty requires external sources and careful calibration to avoid arbitrary “scores.” Sensitivity analysis increases evaluation burden and UI complexity; v1 keeps positioning minimal-but-solid so core scores remain deterministic, interpretable, and stable before layering second-order analytics.

---

## Chapter 4 — Job Intelligence Engine (Recommendations & Optimisation)

### Candidate improvements (headline list)
- **Skill ROI model** (lift ÷ learning cost) once difficulty/cost is defensible.
- **Career path optimisation** (Pareto skill bundles; multi-objective trade-offs).
- **Integrate Chapter 2 signals into ranking** (surgically; measurable uplift only).
- **Career simulation (manual what-if)** as a product feature (currently exists but is de-scoped in v1).
- **Cross-location optimisation** (salary × competitiveness × difficulty; higher-stakes decision support).

### Why these are deferred (explicit rationale)

**1) Career simulation (manual what-if skill additions)**  
- **Why not in v1:** without strong skill normalisation + UI constraints, user-entered tokens often become **no-ops** (OOV/synonyms/typos), which feels broken even when correct.  
- **Why redundant in v1:** the upskilling recommender already runs counterfactuals anchored to the user’s **actual stretch frontier** (derived missing families) and is guardrailed against harming best-now.  
- **V2 trigger:** taxonomy/normalisation upgrades + scenario-builder UI that constrains inputs to valid canonical skills.

**2) Cross-location optimisation (salary × competitiveness × difficulty)**  
- **Why not in v1:** changes the promise from “recommend within constraints” to “recommend where to move,” which is higher-stakes. Requires defensible difficulty/cost, and ideally cost-of-living/relocation comparability, to avoid misleading confidence.  
- **What v1 does instead:** supports relaxing geo constraints, but avoids claiming an optimisation across markets.

**3) Skill ROI (salary uplift ÷ difficulty)**  
- **Why not in v1:** true ROI needs (a) difficulty/cost and (b) credible uplift attribution. v1 can estimate *positioning lift*, but difficulty is external and uplift attribution is noisy.  
- **Portfolio risk if forced into v1:** produces a precise-looking number without validated grounding.

**4) Career path optimisation (Pareto skill choices)**  
- **Why not in v1:** combinatorial search + non-trivial objective definition (lift vs risk vs time vs uncertainty). Requires careful UX and stronger modelling assumptions.  
- **What v1 does instead:** ranked top-k upskilling with clear guardrails.

**5) Integrate Chapter 2 into ranking**  
- **Why not in v1:** adds opacity risk to a recommender that is intentionally interpretable (fit vs barrier).  
- **V2 approach:** introduce Ch2 signals first as *macro exploration* (already partially done), then integrate into ranking only if it improves outcomes under evaluation.

---

## Chapter 5 — App (V2 ideas / deferred scope)

These items were intentionally de-scoped from v1 to keep the app compact and decision-oriented. Add them only if they improve understanding or materially change decisions.

### Market structure (advanced, opt-in)
- **Skill specialisation / lift maps** (Ch2) as an *advanced* panel.  
  - Guardrails: clear labels, hard caps, “how to read” box, and filtering to user-relevant groups.
- **Adjacent job families explorer** (only if families are labelable).  
  - Show only user-conditioned families (e.g., nearest to dominant stretch family) with 2–3 exemplar jobs each.
- **Recommendation concentration / diversity diagnostics** (if families exist and are interpretable).  
  - Optional: Herfindahl-style concentration over state/sector/title when families remain unlabelled.

### Explainability expansions (only if high-signal)
- **SHAP dependence plots** for 2–3 features max, only when they clarify a non-obvious pattern.  
  - Default view remains global SHAP + beeswarm; dependence lives behind expanders.

### Counterfactual / future-facing layer (advanced)
- **Expose career simulation** as explicitly opt-in “manual scenario mode.”  
  - Design: scenario cards (add skills / change constraints / move state) with one “run scenario” action.
  - Principle: never on by default; clearly labelled “counterfactual.”

### Notes on v1 decisions (context)
- v1 avoids unlabelled clusters and dense heatmaps to reduce cognitive load.
- v1 surfaces Chapter 2 mainly through **user-conditioned co-learning neighbours**, the most actionable bridge to upskilling.
- PDP/ICE panels remain out-of-scope because they add complexity without improving the core recommendation story (structure dominates; skills refine within regimes).

---

## Summary principle for v1 → v2 scope

v1 prioritises:
- one canonical recommendation engine,
- one counterfactual engine (upskilling) anchored to stretch jobs,
- deterministic behaviour + evaluators + shipping docs.

v2 items are deferred because they add either:
- **new external dependencies** (difficulty/cost, international comparability, richer normalisation),
- **major UX burden** (scenario building, Pareto interpretation),
- or **feature overlap** that increases complexity without adding new portfolio-grade signal.
