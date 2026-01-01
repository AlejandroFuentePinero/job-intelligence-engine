# Job Intelligence Engine — Project Overview  
Narrative summary of Chapter 0 and the Salary Response Model of Chapter 1  
Date: 2025-12-16

The Job Intelligence Engine converts unstructured job postings into a structured analytical system.  
Its goal is to understand how job titles, skills, industries, and companies shape the labour market, beginning with salary prediction.  
This overview describes the conceptual purpose of Chapter 0 and the first modelling component of Chapter 1.


data source:

https://www.kaggle.com/datasets/andrewmvd/data-scientist-jobs?select=DataScientist.csv
https://www.kaggle.com/datasets/andrewmvd/data-analyst-jobs

---

# Chapter 0 — Purpose & Narrative

Modern job postings are inconsistent: titles vary wildly, salaries are ambiguous, descriptions are noisy, and skills appear in unstructured text.  
Chapter 0 resolves this by transforming heterogeneous raw data into a **clean, standardised, modelling-ready dataset**.

Its core aims:

### **1. Unify heterogeneous data sources**  
Data Analyst and Data Scientist postings are aligned into a common schema with consistent structure.

### **2. Standardise role identity**  
Raw titles are mapped into meaningful abstractions — job family, seniority, and normalised role labels.

### **3. Extract semantic structure**  
Descriptions are cleaned, role domains are assigned, and a curated skill taxonomy is used to build a multi-hot skill representation.

### **4. Normalise salary information**  
Irregular salary formats are converted into annualised numeric fields suitable for modelling.

### **5. Create a reproducible feature space**  
Everything produced in Chapter 0 forms the fixed foundation for all downstream analysis and models in the Job Intelligence Engine.

---

# Chapter 1
# Salary Response Model

Chapter 1 begins the analytical phase of the project by modelling salary as a function of job attributes and skill patterns.

Its guiding ideas:

### **1. Skills require dimensionality reduction**  
Binary skills are sparse and correlated; reducing them into broader latent dimensions improves stability and interpretability.

### **2. Categorical attributes contain structured salary signals**  
Features such as company characteristics, location, job family, and seniority consistently influence observed salary variation.

### **3. Use robust models designed for mixed feature types**  
A boosted decision-tree model is used to learn non-linear interactions between job attributes and salary.

### **4. Build for integration with later chapters**  
The model is wrapped in a clean prediction interface that will support:  
- fairness and residual analysis  
- skill-demand modelling  
- job embeddings  
- recommendation systems  
- competitiveness scoring  

---

# 1.3 Salary Fairness Analysis

A key interpretability component of Chapter 1 is the fairness analysis, which examines how salary residuals vary across major categorical job attributes.  
Using the fitted Salary Response Model, residuals (observed minus predicted salary) are summarised for each category within:

- Location (state)  
- Sector  
- Job title (enriched title representation)  
- Company size  
- Ownership type  
- Seniority level  

For every feature, we compute mean residuals, median residuals, counts, and size-weighted residual means.  
These summaries reveal which groups tend to pay above or below model expectations once all other variables have been accounted for.  
Bar plots provide a visual comparison, and each feature-level table is exported as a separate CSV for modular analysis.  

The fairness results serve as a descriptive diagnostic of structural pay patterns:  
they highlight consistent over- and under-payment trends across the market, informed by the model’s learned structure rather than raw salary averages.

---


---

# 1.4 Skill Requirement Models & Probability Matrix

Following the Salary Response Model, Chapter 1 adds a second analytical layer:  
**estimating the probability that each of the 27 curated skill groups is required for a given job**.  
This component provides the foundation for understanding labour-market skill structure and powers multiple downstream chapters.

---

## Skill Requirement Models (27 binary classifiers)

Each skill group is modelled using a dedicated LightGBM binary classifier.  
For a given target skill, the model uses the following predictors:

- Company attributes (state, sector, size, ownership)  
- Role attributes (seniority, enriched title representation)  
- The remaining 26 skill indicators  

Each model estimates:

**P(skill_k = 1 | job features)**

This setup captures how job attributes jointly influence the likelihood of a skill being required.  
All models are evaluated consistently using ROC AUC, PR AUC, Brier score, calibration plots, and feature importance analysis.

---

## Evaluation Summary

Across the 27 skill models, evaluation metrics were strong and stable:

- **ROC AUC values** were consistently high, reflecting strong ranking performance across all skill groups.  
- **PR AUC values** aligned closely with skill prevalence, performing particularly well for moderate-frequency skills and remaining appropriate for rarer ones.  
- **Brier scores** indicated well-calibrated probability estimates, with error levels consistent with professional-grade binary classifiers.

Feature importance distributions further showed coherent patterns:  
company-level attributes (state, sector) and enriched title signals contributed meaningfully, while co-occurring skills provided much of the predictive structure.  
This confirms that the system is learning real, interpretable relationships between job attributes and skill requirements.

---

## Skill Probability Matrix

Once all models are trained, their outputs are assembled into a full job × skill probability matrix.  
For every job in the dataset, the system computes:

**P(skill_k = required | job attributes)**

This matrix provides a continuous, smoothed representation of skill demand, correcting for noise and sparsity in the raw 0/1 indicators.  
It becomes the **canonical skill layer** for future chapters, enabling:

- job and skill embeddings  
- skill-demand landscapes  
- competency clustering  
- personalised upskilling recommendations  
- labour-market heatmaps  
- downstream explainability analyses  

The matrix is constructed deterministically from the stored models, ensuring reproducibility and consistency across analysis stages.

---

## Conclusion

The skill requirement models and the resulting probability matrix complete the second modelling pillar of Chapter 1.  
They offer a statistically sound, interpretable, and reusable representation of the labour-market skill structure, forming the analytical bridge to all subsequent chapters of the Job Intelligence Engine.


---


# 1.2 Salary Model Explainability (SHAP, PDP & ICE)

In addition to predictive accuracy and fairness diagnostics, Chapter 1 includes a dedicated explainability analysis to understand how the Salary Response Model constructs its predictions. Three complementary tools are used: **SHAP** for feature attribution, **Partial Dependence Plots (PDP)** for average marginal response shapes, and **Individual Conditional Expectation (ICE)** for assessing heterogeneity around those averages.

SHAP (SHapley Additive exPlanations) decomposes each predicted salary into additive contributions from individual features, revealing which job attributes systematically increase or decrease predicted pay. The SHAP analysis shows that salary predictions are driven primarily by **structural labour-market factors**—notably enriched job title, geographic location, sector, and company characteristics—while skill composition influences predictions mainly through **threshold-like effects** rather than smooth marginal gains. Several categorical variables, such as explicit seniority encoding, exhibit near-zero SHAP impact, indicating that their information is largely absorbed by higher-resolution features such as enriched title representation and aggregated skill structure.

To complement SHAP, PDPs are used to characterise the **average functional relationship** between selected continuous skill dimensions and predicted salary. PDPs are computed by fixing a skill principal component to a sequence of values within its central empirical range and averaging model predictions across all observed job contexts. The resulting curves exhibit stepwise behaviour and saturation effects, consistent with the split-based structure of the XGBoost model and closely aligned with the dependence patterns observed in SHAP analyses.

ICE analysis extends this perspective by examining **job-level response trajectories** around the PDP averages. Using a subsample of jobs, ICE curves are computed by varying a single skill component across the same grid of values while holding all other job attributes fixed at their observed levels. By centring predictions relative to a baseline skill value, ICE isolates the marginal effect of skill variation and removes dominant baseline salary differences across jobs. The resulting curves show limited divergence and largely parallel trajectories, indicating minimal interaction-driven heterogeneity and confirming that PDPs provide a faithful summary of the model’s global behaviour.

Taken together, SHAP, PDP, and ICE provide a coherent and internally consistent explanation of the Salary Response Model. SHAP identifies **which features matter and in which direction**, PDPs describe **how predicted salary changes on average** as skill dimensions vary, and ICE validates that these average relationships are **stable across job contexts**. All three analyses are explicitly descriptive, characterising model-implied labour-market structure rather than making causal claims about individual skills or roles.


# 1.5 Global Skill Value Index — Overview

In addition to predicting salary and modelling skill demand, Chapter 1 includes a **global skill value analysis** to interpret how individual skills relate to predicted pay levels within the learned salary structure.  
Rather than treating skills independently, the analysis leverages the model’s latent skill representation to assess how different skills contribute, in aggregate, to higher or lower salary predictions.

The resulting Global Skill Value Index provides a **relative ranking of skills** based on their association with higher predicted salaries, conditional on job role, company characteristics, location, and overall skill composition.  
This index is descriptive and model-implied: it reflects patterns learned by the salary model rather than causal effects or market prescriptions.

The skill value analysis serves as an **interpretability bridge** between the salary model and later chapters, helping contextualise how the model values different skill signals while remaining intentionally lightweight and non-mechanical within the system.


# Chapter 2 — Hidden Structure from Job–Skill Graphs  
Narrative overview of graph construction, embeddings, job families, and skill ecosystems  
Date: 2025-12-15

Chapter 2 shifts the Job Intelligence Engine from feature-based modelling to **structural discovery**.  
Rather than asking how individual variables influence outcomes, this chapter asks how jobs and skills are organised *relative to one another* within the labour market.  
It does so by representing the market as a graph, learning embeddings from that graph, and converting those embeddings into interpretable structure.

---

## From Tabular Features to Relational Structure

The outputs of Chapter 1—clean job attributes and a continuous job × skill probability matrix—implicitly describe relationships between jobs and skills, but those relationships remain distributed across columns and models. Chapter 2 makes this structure explicit by reframing the system as a **job–skill bipartite graph**, where jobs and skills are treated as nodes and edges encode skill demand relationships.

This graph representation captures higher-order dependencies that are difficult to express in tabular form. Jobs are no longer defined only by their attributes, but by their position within a network of shared skill requirements. Likewise, skills are situated within the broader ecosystem of jobs that demand them.

---

## Learning Continuous Representations with Node2Vec

To extract usable structure from the job–skill graph, Node2Vec is applied to learn low-dimensional vector embeddings for both jobs and skills. These embeddings encode each node’s neighbourhood and connectivity patterns within the graph, such that jobs requiring similar constellations of skills are embedded close together, and skills that tend to appear in similar job contexts occupy nearby positions.

The resulting embeddings form a **continuous geometric representation** of the labour market. They are not categories or predictions, but a latent space that preserves relational similarity and supports multiple downstream analyses.

---

## From Embeddings to Interpretable Job Families

While embeddings provide a powerful representation, they are not directly interpretable or reusable as system-level artefacts. To move from geometry to structure, job embeddings are clustered to identify **latent job families**—groups of jobs that occupy similar regions of the embedding space and therefore share comparable skill contexts.

Job embeddings are first L2-normalised so that Euclidean distance reflects directional similarity rather than vector magnitude. KMeans clustering is then applied, with the number of clusters selected using the silhouette score as a stability heuristic. The silhouette curve exhibits a broad plateau rather than a sharp optimum, indicating that job structure is continuous rather than discretely separable. A value of *k = 20* is chosen as a pragmatic balance between resolution and interpretability.

The output of this step is a single, stable artefact mapping each `job_id` to a `job_family_id`. These job families are unsupervised and provisional by design: they are not intended as a definitive taxonomy, but as a reusable structural layer that supports aggregation, comparison, and navigation across the job market.

---

## Skill Ecosystems and Connectivity Structure

In contrast to jobs, skills are not clustered into discrete groups. Skills are inherently overlapping and relational, and forcing them into hard clusters would obscure important structure. Instead, skill embeddings are used to construct a **skill ecosystem network**.

Skill embeddings are L2-normalised and compared using cosine similarity (via dot products). For each skill, only the top-*k* most similar neighbouring skills are retained, producing a sparse, undirected skill–skill network that captures the strongest associations in embedding space. Self-edges are removed and symmetric duplicates are collapsed to yield a clean edge list representation.

This skill ecosystem highlights **skill bundles and co-occurrence structure**: which skills tend to appear together across jobs, which act as bridges between domains, and which occupy more specialised or peripheral positions. The resulting network is an interpretable structural artefact rather than a predictive model, designed to support exploration, visualisation, and downstream reasoning about skill relationships.

---

## Chapter 2 in Context

By the end of this stage, the Job Intelligence Engine has progressed from:
- raw postings (Chapter 0),  
to feature-based modelling of salary and skill demand (Chapter 1),  
to an explicit **structural representation of the labour market** (Chapter 2).

Jobs are now embedded, clustered into families, and indexed within a learned latent structure.  
Skills are embedded and organised into a sparse ecosystem capturing proximity and co-occurrence patterns.

Together, these artefacts form the backbone for subsequent chapters, including industry and domain specialisation, competitiveness scoring, career path simulation, and personalised job and skill recommendations. The emphasis throughout Chapter 2 remains on **structural discovery rather than optimisation**: learning how the market is organised before attempting to navigate or optimise within it.

---

Chapter 2 shifts the Job Intelligence Engine from feature-based modelling to **structural discovery**.  
Rather than analysing how individual variables influence outcomes, this chapter focuses on how jobs and skills are organised *relative to one another* within the labour market.

The chapter reframes the outputs of Chapter 1 as a **relational system** by constructing a probability-weighted job–skill bipartite graph. Jobs and skills are treated as nodes, and edges encode model-implied skill demand. This representation makes higher-order structure explicit: jobs are defined by their position within a network of shared skill requirements, and skills are situated within the ecosystem of jobs that demand them.

To extract latent structure from this graph, Node2Vec is applied to learn low-dimensional embeddings for both jobs and skills. These embeddings capture neighbourhood similarity and co-occurrence patterns, producing a continuous geometric representation of the labour market. Job embeddings are then clustered to infer **latent job families**—unsupervised job ecosystems that support aggregation, comparison, and navigation without imposing a predefined taxonomy.

In parallel, skill embeddings are retained to model **skill ecosystems** rather than forcing skills into discrete clusters. Skill similarity graphs and group-level skill specialisation maps are constructed to identify skill bundles, gateway skills, and over- or under-representation patterns across sectors, titles, seniority levels, ownership types, locations, and job families.

By the end of Chapter 2, the Job Intelligence Engine has progressed from tabular features to a reusable **structural layer**: a graph-derived representation of how jobs and skills relate across the market. This layer forms the backbone for subsequent chapters, enabling ecosystem analysis, specialisation summaries, and recommendation-oriented modules built on learned market structure rather than raw feature space.

# Chapter 3 — Individual Positioning  
Narrative overview of user-level job ranking, skill gaps, and competitiveness  
Date: 2025-12-18

Chapter 3 introduces the **individual-facing layer** of the Job Intelligence Engine.  
Where Chapters 0–2 focus on cleaning data, modelling labour-market structure, and discovering latent organisation, Chapter 3 asks a different question:

**Given a specific user, how does the market look from their position?**

This chapter reframes all upstream artefacts—skills, PCA space, salary structure, and skill-demand probabilities—around a single individual, producing ranked job recommendations, interpretable skill gaps, and diagnostics of access difficulty.

---

## Core Idea

Jobs are not evaluated in isolation.  
They are evaluated **relative to a user’s skills, preferences, and constraints**, using a fixed, deterministic market representation learned earlier in the project.

Chapter 3 separates this evaluation into two complementary dimensions:

- **Suitability** — *How well does this job match the user?*  
- **Competitiveness** — *How hard would this job be for the user to access?*

Both are computed transparently, using calibrated probabilities and explicit assumptions, rather than opaque end-to-end optimisation.

---

## User Profile as the Single Entry Point

The chapter begins by formalising how a person enters the system through a **UserProfile schema**.  
Free-text skill descriptions, location preferences, and role filters are validated, normalised, and projected into the **same 27-skill and PCA space** used by job models.

This guarantees that all comparisons between users and jobs occur in a shared, coherent feature space, and that no downstream module reimplements user parsing logic.

---

## Candidate Set Construction

Before scoring, the system applies **hard constraints** to define the feasible job set:
- location,
- sector,
- enriched title,
- job family.

This step is intentionally strict and deterministic.  
If constraints yield no candidates, the system fails loudly, forcing constraint revision rather than silently degrading recommendations.

---

## Suitability: Matching the User to Jobs

Suitability measures **how well a job aligns with the user’s current profile**, not how prestigious or demanding it is.

It combines:
- **Skill similarity**, computed as cosine similarity between user and job representations in PCA skill space, and  
- **Salary alignment**, formulated as a one-sided score that does not penalise jobs exceeding the user’s stated target.

These components are normalised to a common scale and aggregated using explicit weights.  
Suitability answers: *“If access were free, how good of a fit is this job?”*

---

## Skill Gap Analysis

Suitability alone is insufficient without explanation.  
To support actionable insight, Chapter 3 computes **probability-based skill gaps** using the skill probability matrix.

Rather than relying on binary skill flags, the system estimates:
> *How strongly each skill is required, on average, across the user’s top-K most suitable jobs.*

Skills the user lacks are assigned gap severity proportional to their predicted importance, producing a ranked, calibrated gap profile that supports upskilling decisions and diagnostics.

---

## Competitiveness: Barrier-to-Entry Scoring

Competitiveness captures **how difficult it is for the user to realistically access a job**, independent of desirability.

It combines:
1. **Expected missing skill burden**, computed as the probability-weighted sum of required skills the user lacks.  
2. **Salary percentile**, measuring how demanding the job is relative to peers.

To avoid treating all missing skills equally, the expected missingness is further **weighted by global skill rarity**.  
Missing a rare, specialised skill counts more heavily than missing a ubiquitous one.

Competitiveness answers: *“Even if this job is attractive, how hard would it be to get?”*

---

## Sensitivity Analysis

Both suitability and competitiveness rely on explicit weighting assumptions.  
Chapter 3 therefore includes **formal sensitivity analyses**, varying component weights across a grid and measuring ranking stability using Spearman rank correlation against a baseline configuration.

This step does not optimise weights.  
Instead, it **tests robustness**, identifying whether rankings are stable to reasonable modelling choices or highly sensitive to subjective assumptions.

---

## Chapter 3 in Context

By the end of Chapter 3, the Job Intelligence Engine can:
- position an individual within the learned labour-market structure,
- rank jobs by fit and access difficulty,
- quantify skill gaps probabilistically,
- and validate robustness of rankings.

This chapter completes the transition from **market-level modelling** to **user-level reasoning**, while remaining fully grounded in the deterministic, interpretable artefacts built in earlier stages.

Subsequent chapters build on this positioning layer to explore salary prediction for individuals, recommendation systems, and dynamic career path simulation.

# Chapter 4 — Recommender Engine (v1)  
Narrative overview of context loading, hybrid recommendation, explanations, and counterfactual upskilling  
Date: 2025-12-29

Chapter 4 moves the Job Intelligence Engine from **positioning** to **decision support**.  
Chapter 3 tells the user where they stand (fit, barrier, gaps). Chapter 4 turns those diagnostics into **actionable outputs**: job shortlists split by accessibility, a user-conditioned salary signal for each role, an inspectable explanation layer, and a counterfactual upskilling module that quantifies “what to learn next” in terms of measurable positioning gains.

---

## From “Ranking” to “Recommendations”

Chapter 3 already produces a constrained candidate universe and two core signals:
- **Suitability**: alignment between the user profile and the job (skills + preferences), and  
- **Competitiveness**: barrier-to-entry given missing skill burden, scarcity/rarity, and constraints.

Chapter 4 treats these as upstream truth and answers:
- Which roles are best opportunities **right now**?
- Which roles are plausible but require meaningful growth (**stretch**)?
- How does the user’s predicted salary compare to the market expectations of those roles?
- Which missing skill families provide the strongest **positioning lift** if acquired?

---

## A Single Canonical Context Payload

To keep notebooks and `src/` aligned and avoid fragile glue code, Chapter 4 standardises around a **context loader** (`load_ch4_context`) that:
1) calls Chapter 3 positioning to produce the user profile and candidate set,  
2) loads aligned artefacts needed for skill reasoning (family probabilities, gap table, etc.), and  
3) loads the persisted Salary Response Model (Chapter 1) and constructs the candidate-level salary feature matrix by broadcasting the user’s skill PC vector across candidate rows.

This yields one “ready-for-recommenders” payload aligned by `job_id`.

---

## Hybrid Job Recommendation: Retrieve → Bucket → Rerank (+ Scored Universe)

The v1 recommender is deliberately simple and deterministic:

1) **Retrieve (suitability gating)**  
   Apply a base suitability threshold; fall back to a lower floor if too few jobs remain; fail loudly if still too small.

2) **Bucket (accessibility)**  
   Split into:
   - **best_now** if `competitiveness_index <= c_max`
   - **stretch** otherwise  
   Bucket-size warnings are emitted rather than silently degrading recommendation quality.

3) **Rerank (within-bucket)**  
   Rank by `score = suitability - alpha * competitiveness_index` to favour fit while penalising barrier.

4) **Scored universe (for counterfactuals)**  
   In addition to the gated candidate set and Top-N tables, Chapter 4 produces a `scored_universe` table containing bucket + score for *all* candidate jobs under the user’s constraints. This stable substrate supports counterfactual analysis (upskilling and optional career simulation) and downstream Chapter 5 reporting.

---

## User Salary Signal: “Market Expected” vs “Skill-Implied”

Each candidate job includes:
- **Market salary** (`sal_mean`): job-side expected salary signal derived from the dataset, and  
- **User-conditioned salary** (`pred_sal`): Chapter 1 model prediction using job codes plus the user’s broadcast skill PCs.

The comparison is an interpretable alignment diagnostic (not a causal claim): whether the user’s skill signal is priced below/near/above the market structure captured in the model.

---

## Explanation Layer (v1): Making Outputs Inspectable

`build_job_explanations()` augments outputs with:
- deterministic “why this bucket / why this rank / salary context” strings, and
- per-job skill-family coverage and gaps:
  - infer required families by thresholding `{family}_prob` at `tau`
  - compare to the user’s family vector to produce `covered_families` and `missing_families`

Critically, the explanator also supports the **full scored universe** (not just Top-N), enabling downstream counterfactual modules to use missing-family fields at scale.

---

## Upskilling (v1): Counterfactual Positioning Gains on a Frozen Universe

The upskilling recommender converts “gaps” into “what to learn next” by simulation:
1) **Freeze the job universe** by `job_id` using `candidate_override_df`, so every upskill scenario compares the same jobs.
2) **Source candidate skill families** primarily from **stretch jobs** (their `missing_families` are the natural targets).
3) **Inject representative tokens** for each missing family into the user skill text and rerun the recommender on the same frozen universe.
4) **Compute per-job deltas** vs baseline (percentage-point deltas for bounded 0–1 metrics).
5) **Rank skill families** by a composite impact score that rewards:
   - promotion rate (stretch → best_now), and
   - score gains (especially on baseline-stretch jobs),
   while penalising demotions and worst-tail harms among baseline best_now jobs (guardrailed via `demotion_tol`).

Outputs are designed to plug directly into Chapter 5: top skill families + example tokens + summary impact metrics + long-table deltas for plots.

---

## Career Simulation (v2 / optional): User-Defined What-If Scenarios

In addition to upskilling recommendations (which auto-select and rank missing families), the same frozen-universe counterfactual machinery can be exposed as a **user-defined simulator**.  
A career simulation module (`career_simulation`) accepts explicit scenarios (e.g., “add SQL + dbt” or “add AWS + Docker”), reruns the recommender on the same frozen job set, and reports:
- scenario-level promotion/demotion rates and mean score deltas, and
- the top “unlocked” jobs that move from `stretch` to `best_now`.

Because v1 uses a curated skill dictionary, the simulator includes **no-op detection**: scenarios are skipped when injected tokens do not change the extracted user skill vector (preventing misleading “fake effects” from out-of-vocabulary tokens). This feature is deferred to v2 because its usefulness depends on broader token coverage and a more robust skill-normalisation layer.

---

## Chapter 4 Scope Boundary (v1)

Chapter 4 v1 delivers a complete decision-support loop:
- canonical context contract,
- deterministic job recommendations with salary signal,
- inspectable explanations including per-job missing families,
- counterfactual upskilling ranked by measurable positioning lift.

Deferred or future enhancements include career simulation as an explicit user-facing tool, richer R
