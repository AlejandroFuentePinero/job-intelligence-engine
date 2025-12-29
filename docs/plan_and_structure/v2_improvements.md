# V2 Improvements Backlog

This backlog captures plausible v2 improvements that expand capability beyond the v1 deliverable.  
v1 is intentionally constrained to a **deterministic, auditable, end-to-end system** with strong contracts and evaluation. v2 items are mostly deferred because they (i) require new external data, (ii) materially increase surface area and testing burden, or (iii) add UX complexity without proportional portfolio value at v1.

---

## Chapter 0 — Foundations (Preprocessing & Taxonomy)
- Job title embeddings and clustering  
- Improve skill dictionary  
- Reinforce salary extraction from description  
- Retrieve experience requirements from job descriptions  
- NER / transformer-based skill extraction (reduce reliance on pure dictionary/regex; improve recall/precision)  
- Domain-specific embedding fine-tuning for title/skill tokens (SBERT fine-tune) if clustering quality is a bottleneck  
- Stronger seniority detection rules (more robust to messy titles; consistent mapping across job families)  
- Entry point for additional data and potential international data (salary normalisation and currency translation)  

**Why deferred (v1 rationale):** These changes shift the entire downstream pipeline by altering the base feature layer. They are high leverage but also high risk: they can invalidate earlier benchmarks and require re-validation across Chapters 1–4. In v1, the priority is to lock a stable taxonomy/extractor contract so positioning + recommendations are reproducible and testable.

---

## Chapter 1 — System Mechanics (Salary, Skill Demand, Fairness)
- Revisit salary model by adding new info and/or changing objectives  
- Estimate uncertainty in skill and salary model  
- Quantile regression / prediction intervals for salary (P10/P50/P90), not just point estimate  
- Stronger segment diagnostics + tougher validation split (e.g., grouped splits by company/title cluster) to stress-test generalization  
- Skill value ranking by city/sector/title (city/sector-specific marginal value rather than global)  

**Why deferred (v1 rationale):** These are “model quality” upgrades that are valuable but can spiral into open-ended iteration. For v1, a point-estimate salary signal is sufficient as an *interpretable alignment cue* in the recommender, and deeper model improvements are better treated as a focused refinement pass after the recommender system is fully closed and evaluated.

---

## Chapter 2 — Hidden Structure (Graphs, Clusters, Ecosystems)
- Alternative clustering method for job families (HDBSCAN) if KMeans clusters are not coherent  
- Industry specialization maps (clustered skill vectors / specialization profiles per cluster or sector)  
- Contrastive job embeddings (Siamese SBERT) if Node2Vec embeddings are too “graph-structure driven” and not semantically clean  

**Why deferred (v1 rationale):** These are “macro-market” enhancements that mainly improve ecosystem interpretability and exploration. v1 only needs stable job-family and skill-similarity artefacts for a minimal macro layer; deeper embedding/clustering alternatives are v2 once you have evidence that the current clusters are not coherent enough for storytelling.

---

## Chapter 3 — Individual Positioning (Suitability, Competitiveness, Gaps)
- Skill rarity integration (inverse frequency weighting in fit / gap severity)  
- Skill difficulty integration (O*NET or proxy difficulty scores) for “gap cost” not just “gap size”  
- Suitability sensitivity analysis (how recommendations change when adding/removing specific skills)  
- City-level competitiveness aggregation (roll job-level competitiveness into city summaries)  

**Why deferred (v1 rationale):** Rarity/difficulty are important, but difficulty requires external sources and careful calibration to avoid arbitrary scores. Sensitivity analysis is useful but expands evaluation burden significantly. v1 keeps positioning minimal-but-solid: the core scores must be deterministic, interpretable, and stable under reasonable settings before adding richer second-order analytics.

---

## Chapter 4 — Job Intelligence Engine (Recommendations & Optimization)
- Skill ROI Model (salary uplift ÷ difficulty)  
- Career Path Optimization (Pareto skill choices)  
- Add ch2 outputs to the recommendation engine  
- Career Simulation (what-if skill additions)  
- Cross-City Optimization (salary × competitiveness × difficulty)  

### Why these Chapter 4 items are deferred (explicit rationale)

**1) Career Simulation (what-if skill additions)**  
- **Why it’s not in v1:** As implemented, it is mostly a generic counterfactual API: the user must guess which skills to test. Without a UI and a stronger skill normalisation layer, many user-proposed tokens become **no-ops** (out-of-vocabulary or typo/synonym issues), which feels broken even when correct.  
- **Why it’s redundant in v1:** The upskilling recommender already provides counterfactual simulation in a more defensible way: it is anchored to the user’s *actual stretch frontier* (missing families derived from stretch jobs), ranked by measured lift, and guardrailed against harming best-now roles.  
- **When it becomes v2-worthy:** After taxonomy/normalisation improves (synonyms/typos/coverage), and/or a scenario-builder UI constrains users to “valid” skills, career simulation can be exposed as a “manual scenario” mode that reuses the same counterfactual engine.

**2) Cross-City Optimization (salary × competitiveness × difficulty)**  
- **Why it’s not in v1:** It changes the product promise from “recommend within constraints” to “recommend where to move,” which is a higher-stakes decision support feature. It also requires a defensible **difficulty** signal (external), plus careful handling of cost-of-living, relocation friction, and comparability across markets to avoid misleading outputs.  
- **What v1 does instead:** v1 supports dropping geo constraints as a simple extension of the recommender, but does not claim a rigorous optimisation across locations because the missing inputs would make it look overconfident.

**3) Skill ROI Model (salary uplift ÷ difficulty)**  
- **Why it’s not in v1:** A true ROI needs (a) difficulty/cost and (b) a credible salary-uplift estimate attributable to that skill change. v1 can estimate *positioning lift* (score changes, promotions), but difficulty is external and salary uplift attribution is noisy.  
- **Risk if forced into v1:** Produces a “precise number” that looks scientific but is not grounded in validated difficulty/cost data—bad for portfolio credibility.

**4) Career Path Optimization (Pareto skill choices)**  
- **Why it’s not in v1:** Multi-skill Pareto optimisation explodes the search space (combinatorics), and the objective is non-trivial: you need trade-offs between lift, risk (demotions), learning cost, time-to-skill, and possibly uncertainty. This demands careful UX (explain Pareto sets), more evaluation, and stronger modelling assumptions than v1 needs.  
- **What v1 does instead:** Provides a ranked top-*k* upskilling list with guardrails. That is simpler, clearer, and interview-friendly.

**5) Add Chapter 2 outputs to the recommendation engine**  
- **Why it’s not in v1:** It is valuable but should be added surgically to avoid turning the recommender into an opaque hybrid. In v1, recommendations are intentionally interpretable (fit vs barrier). Chapter 2 is best introduced first as a macro/adjacent-role exploration layer (Chapter 5 storytelling), then optionally integrated into ranking in v2 once you can evaluate whether it improves outcomes.

---

## Summary principle for v1 → v2 scope
v1 prioritises:
- one canonical recommendation engine,
- one counterfactual engine (upskilling) anchored to stretch jobs,
- deterministic behaviour + evaluators.

The Chapter 4 items above are deferred because they add either:
- **new external dependencies** (difficulty, better taxonomy, richer geo comparability),
- **major UX burden** (scenario building, Pareto interpretation),
- or **feature overlap** that increases complexity without adding new signal for a portfolio-grade v1.
