# V2 Improvements Backlog

## Chapter 0 — Foundations (Preprocessing & Taxonomy)
- Job title embeddings and clustering
- Improve skill dictionary
- Reinforce salary extraction from description
- Retrieve experience requirements from job descriptions
- NER / transformer-based skill extraction (reduce reliance on pure dictionary/regex; improve recall/precision)
- Domain-specific embedding fine-tuning for title/skill tokens (SBERT fine-tune) if clustering quality is a bottleneck
- Stronger seniority detection rules (more robust to messy titles; consistent mapping across job families)
- Entry point for additional data and potential international data (salary normalisation and currency translation)

## Chapter 1 — System Mechanics (Salary, Skill Demand, Fairness)
- Revisit salary model by adding new info and/or changing objectives
- Estimate uncertainty in skill and salary model
- Quantile regression / prediction intervals for salary (P10/P50/P90), not just point estimate
- Stronger segment diagnostics + tougher validation split (e.g., grouped splits by company/title cluster) to stress-test generalization
- Skill value ranking by city/sector/title (city/sector-specific marginal value rather than global)

## Chapter 2 — Hidden Structure (Graphs, Clusters, Ecosystems)
- Alternative clustering method for job families (HDBSCAN) if KMeans clusters are not coherent
- Industry specialization maps (clustered skill vectors / specialization profiles per cluster or sector)
- Contrastive job embeddings (Siamese SBERT) if Node2Vec embeddings are too “graph-structure driven” and not semantically clean

## Chapter 3 — Individual Positioning (Suitability, Competitiveness, Gaps)
- Skill rarity integration (inverse frequency weighting in fit / gap severity)
- Skill difficulty integration (O*NET or proxy difficulty scores) for “gap cost” not just “gap size”
- Suitability sensitivity analysis (how recommendations change when adding/removing specific skills)
- City-level competitiveness aggregation (roll job-level competitiveness into city summaries)

## Chapter 4 - Job Intelligence Engine (Recommendations & Optimization)
- Skill ROI Model (salary uplift ÷ difficulty)
- Career Path Optimization (Pareto skill choices)
- Add ch2 outputs to the recommendation engine
