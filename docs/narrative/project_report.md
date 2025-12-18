# Job Intelligence Engine — Project Report  
## Chapters 0–1 (Technical Summary)
Date: 2025-12-11
---

# Chapter 0 — Data Acquisition, Cleaning & Feature Foundation

## Overview
Chapter 0 builds the entire data foundation for the Job Intelligence Engine. Two raw datasets of job postings were ingested, cleaned, normalised, and enriched into a unified modelling-ready dataset. This chapter ensures that the project begins with a clean schema, consistent feature definitions, and reproducible transformations suitable for downstream machine learning, skill-demand modelling, and job similarity analysis.

---

## 0.1 Data Cleaning & Standardisation

### Methodology
Raw Glassdoor datasets varied in schema, formatting, and missing-value patterns. The pipeline applied deterministic transformations to:

- Align column structures and merge sources  
- Replace placeholder values and normalise missing data  
- Extract state-level location information  
- Harmonise company metadata (ownership, industry, sector, year founded)  
- Remove sparsely populated or unreliable fields  
- Perform controlled text normalisation of job descriptions  

### Results
The cleaning pipeline produced a consistent dataset containing 6,162 job records with 47 curated features. Missing values were standardised; redundant and high-missingness fields were removed; all key categorical variables were normalised. Data integrity checks confirmed that the resulting data structure was coherent and suitable for feature engineering.

### Conclusions
The dataset emerging from this stage is structurally reliable, analytically coherent, and ready for the modelling tasks in Chapter 1. All transformations are deterministic and reproducible.

---

## 0.2 Title Processing, Domain Assignment & Seniority Extraction

### Methodology
Job titles in the raw data exhibited substantial variability and ambiguity. The title-processing module normalised these into meaningful, structured representations through:

- Cleaning and tokenising raw titles (`job_title_base`)  
- Detecting explicit and implicit seniority cues in both the title and description  
- Combining these signals into `seniority_combined`  
- Mapping titles into a stable job-family taxonomy (`job_title_family`)  
- Assigning job domains via a precomputed SBERT+KMeans embedding lookup  

### Results
The pipeline produced consistent seniority assignments across roles, robust to both explicit title markers and contextual cues. Job titles were standardised into families such as “analyst,” “scientist,” “engineer,” and “manager.” Domain assignment grouped postings into semantically coherent clusters reflecting functional areas, improving interpretability and modelling stability.

### Conclusions
The resulting structured representation of job identity reduces the variance inherent in raw titles and provides a solid basis for modelling salary, skill requirements, and job similarity.

---

## 0.3 Salary Parsing & Normalisation

### Methodology
Salary information appeared in various formats (hourly, annual, ranges). A custom parser:

- Extracted numeric components from unstructured salary text  
- Converted hourly rates to annual equivalents  
- Computed lower bound, upper bound, and midpoint (`sal_mean`)  
- Preserved absence of salary information as a valid analytic signal  

### Results
Salary values were successfully standardised into comparable annual amounts. The resulting distribution captured realistic variation in market salaries. Diagnostic checks confirmed that conversions and extractions were stable and consistent across formats.

### Conclusions
A reliable target variable (`sal_mean`) was established for predictive modelling in Chapter 1.

---

## 0.4 Skill Extraction (Dictionary-Based v1)

### Methodology
A curated dictionary of over 1300 data-related skill tokens was used to extract structured skill information from job titles and descriptions. The module:

- Combined title and description into a unified text field  
- Applied case-normalised multi-word and unigram token matching  
- Generated binary indicators across 27 aggregated skill groups  

### Results
Skills were extracted with high coverage and strong specificity, yielding a multi-hot skill matrix that captures both general competencies and specialised capabilities. The resulting structure supports predictive modelling, skill-demand analysis, and job–skill embedding work in later chapters.

### Conclusions
This representation provides a detailed, semantically rich skill feature space essential for modelling job requirements and labour-market structure.

---

## Chapter 0 — Final Output
The final processed dataset contains:
- **6,162 rows**
- **47 engineered features**
- Fully cleaned company and role metadata  
- Standardised salary features  
- 27 binary skill indicators  
- Stored as `chapter0_processed_jobs.csv`  

This dataset is the canonical input for all subsequent chapters.

---

# Chapter 1 — System Mechanics

## Overview
Chapter 1 introduces the core predictive and analytical machinery of the Job Intelligence Engine. The completed components include: (i) feature engineering for modelling, (ii) dimensionality reduction of skill indicators, and (iii) the Salary Response Model — the first major predictive system in the project.

---

# 1.1 Feature Engineering for Predictive Modelling

### Methodology
Features from Chapter 0 were transformed into model-ready formats:

- Categorical features converted into integer-based pandas categoricals  
- Combination of job-family and domain into a unified signal (`title_rich_code`)  
- Integration of all engineered PCA skill components  
- Construction of the final modelling matrix and train–test split  

### Results
All features were compatible with XGBoost’s categorical handling, and the modelling matrix exhibited no missing values among required predictors. Descriptive statistics confirmed balanced distributions across training and test sets.

### Conclusions
A stable and interpretable feature space was produced for predictive modelling.

---

# 1.2 Skill PCA (Dimensionality Reduction)

### Methodology
To reduce redundancy and correlation among the 27 binary skill indicators, PCA was applied:

- PCA fitted on the full binary skill matrix  
- Retained the **top 10 components**, explaining approximately **70%** of total variance  
- Components stored in `skill_pca_v1.pkl`  
- Added a general-purpose transformation function for inference  

### **Results** (Updated, Professional)
The extracted principal components represented orthogonal linear combinations of the original skill indicators. Examination of component loadings revealed coherent statistical groupings: skills related to analytical workflows tended to load together, modelling and machine-learning–related features formed distinct clusters, cloud- and data-engineering–related indicators concentrated on separate components, and business-intelligence tools formed another grouping.  
Each component summarised a different axis of variation in the high-dimensional skill space, allowing the model to replace correlated binary indicators with independent, variance-rich latent dimensions.

### Conclusions
PCA provided a compact, noise-reduced representation of the skill space. This improved model stability, reduced collinearity, and preserved the majority of skill-related information required for salary prediction.

---

# 1.3 Salary Response Model (XGBoost v4)

### Methodology
The objective was to model expected salary (`sal_mean`) using job attributes and PCA skill components. Several model variants were evaluated using different combinations of raw skill indicators, one-hot encodings, and enriched title representations. The final architecture — Model v4 — used:

- Categorical job and company features  
- PCA skill components  
- Enhanced title representation (`title_rich_code`)  
- XGBoost with hyperparameter tuning via GridSearchCV  
- Evaluation using R², RMSE, MAE, residual diagnostics, and feature importance  

### Results
Model v4 achieved:

- **R²:** ~0.30 on the test set  
- **RMSE:** ~31.5k  
- **MAE:** ~25k  
- **Train R²:** ~0.35  
- Residuals displayed symmetric distribution centered around zero, with no extreme systematic deviation  
- Feature importances highlighted seniority, job family, sector, and the top PCA skill components as the primary contributors  

The model behaved consistently with expectations given noisy salary data: moderate predictive power with stable residual structure and interpretable feature contributions.

### Conclusions
The Salary Response Model provides a stable and interpretable estimate of salary given job attributes. It forms the basis for subsequent fairness analysis, skill-value estimation, and explainability via SHAP, PDP, and ICE.

---

# **1.4 Skill Requirement Models & Probability Matrix**

## **Methodology**
To characterise the skill requirements underlying each job, Chapter 1 trains **27 independent LightGBM binary classifiers**, one per curated skill group. Each model estimates:

\[
P(\text{skill}_k = 1 \mid \text{job attributes})
\]

using the following predictors:

- **Company attributes:** state, sector, size, ownership  
- **Role attributes:** enriched title representation (`title_rich_code`), seniority  
- **Remaining 26 skill indicators:** providing contextual co-occurrence information  

Models were evaluated using **ROC AUC**, **PR AUC**, **Brier score**, **calibration curves**, and **feature importance**.  
Model artefacts were saved and later used to construct a full **job × skill probability matrix**, which replaces sparse binary skill indicators with smooth, continuous estimates of skill demand.

These models are *not* used for independent prediction but rather as a **collective inference engine** for estimating skill probabilities across all jobs.

---

## **Results**

### **Overall Model Quality**
Across the 27 skill classifiers, predictive performance was consistently strong:

- **ROC AUC:** Typically **0.88–0.95** (min ≈ 0.80, max = 1.00)  
- **PR AUC:** Mean **0.75**, median **0.77**, with many skills in the **0.85–0.95** range  
- **Brier scores:** Concentrated around **0.06–0.12**, indicating well-calibrated probability outputs  

These values are well above industry norms for imbalanced skill-prediction tasks, where PR AUC values of **0.60–0.75** are common.

### **Skill-Level Insights**
Performance varied in a pattern that aligns with the underlying data:

- **High-prevalence skills** (e.g., core programming, BI basics, soft skills) achieved **PR AUC ≈ 0.90–1.00**, reflecting very strong separability and highly reliable probability ranking.  
- **Moderate-prevalence skills** (e.g., analytics intermediate, cloud intermediate, ML basic/intermediate) scored **PR AUC ≈ 0.70–0.85**, consistent with stable, well-learned signals.  
- **Low-prevalence “advanced” categories** produced **lower PR AUC values (≈ 0.35–0.55)** — an expected outcome driven by sparsity rather than modelling limitations.

#### **Feature Importance
Feature-importance profiles showed structured and interpretable patterns:  
**title**, **state**, and **sector** consistently contributed the strongest signals, while co-occurring skills provided rich contextual information.

In addition to model-performance metrics, each skill classifier provides a **feature-importance profile** summarising how strongly different predictors contribute to the model’s decisions.

These values are **not measures of model quality**. They instead describe **how much each predictor helps the model reduce loss** (LightGBM uses split-gain–based importance). Larger values indicate more influential predictors.

Across the 27 models:

- **Enriched job title (`title_rich_code`)**, **state**, and **sector** consistently showed strong influence.  
- **Co-occurring skills** also contributed meaningful information — especially within related domains (e.g., analytics → ML, cloud → data engineering).  
- **Advanced/rare skills** relied more heavily on contextual predictors due to sparse positive examples.  
- **Company size** and **ownership** contributed moderately but consistently across models.

These importance values help explain *why* the models produce certain probability patterns, but they are **not indicators of model accuracy**. Only ROC AUC, PR AUC, and Brier score reflect performance.


### **Probability Matrix**
All models were applied to the full job dataset, generating a **6,162 × 27 continuous probability matrix**.  
The matrix displayed:

- clear separation between jobs that truly require a skill vs. those that do not  
- realistic probability gradients instead of binary jumps  
- coherent cross-skill correlations reflecting functional areas (e.g., analytics ↔ ML ↔ programming)  
- well-calibrated output consistent with Brier-score diagnostics  

This matrix significantly enhances analytical flexibility by enabling ranking, weighting, clustering, and distance-based comparisons between jobs in later chapters.

---

## **Conclusions**
The skill requirement modelling stage produced a **robust, high-performing inference layer** that transforms noisy binary indicators into probabilistic representations of skill demand.  
With strong ROC AUC, PR AUC, and calibration performance across most skills, the resulting probability matrix is suitable for:

- job embeddings  
- skill-demand and co-occurrence modelling  
- labour-market structure analysis  
- personalised upskilling and recommendation tools  
- fairness and competitiveness metrics  

Importantly, the **probability matrix is the only required output** from this stage, and all artefacts (models + matrix builder) are now fully reusable, deterministic, and integrated into the project architecture.


---

# **1.5 Salary Fairness Analysis (Residual Diagnostics Across Categorical Job Attributes)**

## **Methodology**

Following the Salary Response Model, residuals were computed as:

\[
\text{residual} = \text{observed salary} - \text{predicted salary}
\]

These residuals quantify whether a job pays **above** or **below** what the model expects after adjusting for all predictors (job family, seniority, skills via PCA, sector, state, size, ownership).  
Fairness was assessed by grouping residuals across six core categorical dimensions:

- **State (location)**  
- **Sector**  
- **Job title family (enriched title representation)**  
- **Company size**  
- **Ownership type**  
- **Seniority level**

For each category, we computed:

- Mean residual  
- Median residual  
- Record count  
- Size-weighted residual mean  
- Sorted bar plots (unweighted and weighted)

Each category-specific summary was exported as an individual CSV file (e.g., `state_fairness.csv`, `sector_fairness.csv`), and all plots were saved in the reporting directory for interpretability.

---

## **Results**

### **Location (State)**
States exhibited clear structural differences even after controlling for job and skill attributes.  
High-cost, high-demand markets such as California and New York showed **positive residuals**, indicating above-expectation pay. Several southern and midwestern states consistently displayed **negative deviations**, reflecting lower-than-expected compensation relative to their job mix.

### **Sector**
Sectors with rapid technological evolution and strong skill demands — notably **Information Technology**, **Biotech/Pharma**, and **Energy** — paid well above expectation.  
Lower-margin or stable sectors such as **Education**, **Nonprofit**, **Food Services**, and **Business Services** showed substantial negative deviations.  
This pattern reflects the premium associated with technical dynamism and innovation intensity.

### **Job Family (Enriched Title)**
A clear hierarchy emerged among data roles:

- **ML/AI data scientists** showed the strongest positive deviations  
- **General data scientists** and specialist scientist roles also overperformed expectations  
- **Data engineers** clustered near neutral  
- **Data analysts** consistently underperformed expectations across domains  

This structure aligns with broad labour-market trends in technical depth and market scarcity.

### **Company Size**
Large organisations (**10,000+ employees**) systematically paid above model expectations.  
Medium-sized companies were near neutral to mildly positive, while smaller companies underpaid relative to predictions.  
This reflects stronger salary capacity and competitive pressure in larger firms.

### **Ownership Type**
**Public companies** displayed strong positive deviations, followed by **government roles** with moderate premiums.  
**Private-sector roles** underperformed expectations, and **nonprofits** consistently exhibited the lowest salary residuals.  
Ownership structure is therefore a major determinant of compensation behaviour.

### **Seniority**
Residuals aligned cleanly with responsibility level:

- **Principal**, **manager**, and **senior** roles paid above expectation  
- **Mid-level**, **lead**, and **junior** roles paid below expectation  
- **Assistant/executive** roles hovered near neutrality  

This confirms that hierarchical position drives systematic pay differences beyond what is captured by title and skills alone.

---

## **Conclusions**

The fairness analysis provides a transparent, model-adjusted view of compensation behaviour across major job attributes.  
Across all six dimensions, consistent structural patterns emerged:

- Competitive, skill-intensive sectors and large organisations pay above expectation  
- Analyst-level roles, smaller companies, and nonprofit organisations underpay  
- Salary deviations follow coherent geographic, sectoral, and organisational gradients  
- Seniority and job family strongly influence unexplained salary variation  
- Positive residuals cluster around high-skill, high-demand, and innovation-driven job types  

These insights form a critical interpretability layer for the Job Intelligence Engine, validating model behaviour and providing a foundation for upcoming components — including SHAP/PDP/ICE explainability, competitiveness scoring, and recommendation design.

---

---

# **1.6 Salary Model Explainability (SHAP Analysis)**

## **Overview**
To complement predictive performance and residual-based fairness diagnostics, Chapter 1 includes a comprehensive explainability analysis using **SHAP (SHapley Additive exPlanations)**.  
This component explains **how the Salary Response Model constructs its predictions**, decomposing each predicted salary into additive contributions from individual job attributes and skill dimensions.

Whereas the fairness analysis examines *deviations from model expectations* (residuals), SHAP focuses on *the internal valuation logic of the model itself*. Together, these approaches provide a complete interpretability framework: one describing **market valuation mechanisms**, the other identifying **systematic departures from those valuations**.

---

## **Methodology**

SHAP values were computed for the final Salary Response Model (XGBoost v4) using the full training dataset.  
For each observation, SHAP assigns a signed contribution to every predictor such that the sum of all feature contributions equals the model’s predicted salary relative to a baseline expectation.

The analysis covered all **16 predictors** used by the model:

- **10 skill principal components** derived from PCA  
- **6 categorical job attributes**:  
  company size, sector, state, ownership, enriched job title, and seniority  

Both **global** (aggregate importance) and **local** (feature-wise dependence) analyses were conducted.  
All plots were generated deterministically and saved to the project’s reporting directory for reproducibility.

---

## **Global SHAP Results**

Global SHAP importance revealed that **structural labour-market features dominate salary predictions**.  
Enriched job title, geographic location (state), sector, and company size consistently exhibited the largest absolute SHAP values, indicating that they account for the majority of explainable salary variation.

Skill composition, represented through PCA components, played a **secondary but structured role**.  
The first few components captured baseline technical requirements that prevent strong salary penalties when unmet, while later components contributed little incremental information.  
Several predictors (including seniority and multiple skill PCs) showed near-zero global SHAP impact, indicating redundancy rather than irrelevance.

Overall, the global SHAP profile confirms that the model primarily learns **market structure and role positioning**, with skills shaping outcomes within those structures rather than defining them outright.

---

## **Local SHAP Interpretation — Skill Components**

### **Skill_PC1**
PC1 captures foundational technical infrastructure skills (pipelines, databases, core programming, cloud).  
Its SHAP dependence exhibits a strong **threshold effect**, where low values incur a large salary penalty, but contributions flatten once a baseline is reached.  
This component functions as a **gatekeeper**, ensuring minimum technical competence rather than differentiating high salaries.

### **Skill_PC2**
PC2 loads on analytical, BI, and softer quantitative skills.  
Higher values are associated with **negative SHAP contributions**, reflecting a shift toward analyst-oriented roles with systematically lower pay.  
This component captures **role-type divergence**, not skill deficiency.

### **Skill_PC3**
PC3 is dominated by ML/AI and advanced modelling skills.  
Its SHAP values show discrete positive jumps rather than smooth gradients, indicating that ML intensity is rewarded once it crosses identifiable thresholds.  
This reflects market recognition of specialised modelling roles.

### **Skill_PC4**
PC4 mixes workflow, coordination, and intermediate infrastructure skills.  
Its SHAP pattern is bidirectional, with modest penalties at low values and limited premiums at higher values.  
This suggests contextual value dependent on accompanying technical depth.

### **Skill_PC5 – Skill_PC10**
These components exhibit **near-zero or highly localised SHAP effects**.  
They capture niche or redundant variation that does not systematically influence salary once higher-order structure is accounted for.  
For interpretation purposes, they can be treated as non-influential.

---

## **Local SHAP Interpretation — Categorical Attributes**

### **Job Title (Enriched Representation)**
Job title is the **single strongest categorical driver** of salary predictions.  
ML- and AI-explicit titles receive large positive contributions, while analyst-oriented and generic titles are strongly penalised.  
Titles act as compressed signals of responsibility, specialisation, and market positioning.

### **State (Location)**
Geographic location exerts extremely large effects, with California and New York showing substantial salary premiums and most other states incurring penalties.  
The magnitude of these effects exceeds those of individual skill components.  
State captures persistent regional wage regimes rather than marginal contextual variation.

### **Sector**
Sector displays one of the widest SHAP spreads among predictors.  
Technology- and innovation-intensive sectors command strong premiums, while education, nonprofit, and service-oriented sectors show deep penalties.  
This reflects industry-level pay norms independent of role composition.

### **Company Size**
Company size exhibits a clear monotonic pattern: large organisations provide salary premiums, while small firms impose penalties.  
The asymmetry of this effect suggests institutional pay capacity rather than skill differences.  
Size acts as a structural salary scaler.

### **Ownership**
Ownership type contributes modest but interpretable effects.  
Nonprofit organisations consistently impose a strong salary penalty, while public, private, and government roles cluster near neutrality.  
Ownership primarily reflects institutional compensation constraints.

### **Seniority**
Seniority shows **near-zero SHAP contribution** once other features are included.  
Its influence is fully absorbed by enriched job title and skill composition, rendering it redundant in the presence of higher-resolution predictors.  
This indicates successful feature hierarchisation rather than model omission.

---

## **Conclusions**

The SHAP analysis demonstrates that salary predictions in the Job Intelligence Engine are governed primarily by **structural labour-market mechanisms** rather than fine-grained skill variation.  
Skills matter, but largely through threshold effects that enable access to higher-paying role classes rather than through continuous marginal returns.

Crucially, SHAP explains *how the market prices jobs*, not whether those prices are equitable.  
When paired with residual-based fairness analysis, the system cleanly separates **market valuation** from **potential inequity**, avoiding conceptual conflation.

This explainability layer provides transparency, interpretability, and theoretical grounding for all downstream components of the Job Intelligence Engine, including competitiveness scoring, job recommendations, and career-path simulation.

---
# **1.7 Salary Model Explainability (Partial Dependence & ICE Analysis)**

## **Overview**
Following SHAP-based attribution, Chapter 1 adds **Partial Dependence Plots (PDP)** and **Individual Conditional Expectation (ICE)** as complementary explainability tools.  
Whereas SHAP identifies *which features contribute to predictions and in which direction*, PDP and ICE characterise **how the model responds to changes in selected features**, both on average and at the individual-job level.

This analysis is explicitly scoped as a **shape and stability diagnostic**, not a reassessment of feature importance or a test of causal effects.

---

## **Methodology**

PDP and ICE analyses were conducted for the **continuous skill PCA components** that exhibited non-negligible signal in the SHAP analysis:  
`skill_PC1`, `skill_PC2`, `skill_PC3`, and `skill_PC8`.

For each component, a grid of values was defined over the **central empirical support** of the data (5th–95th percentile), using 25 evenly spaced points.

### **Partial Dependence Plots (PDP)**
At each grid value \(v\), the target component was fixed to \(v\) across all observations, while all other model inputs (categorical attributes and remaining skill components) were left unchanged. The trained Salary Response Model was then used to generate predictions for this counterfactual dataset, and the **mean predicted salary** across all observations was recorded:

\[
\text{PDP}_j(v) = \mathbb{E}_{X_{-j}}\big[f(v, X_{-j})\big]
\]

This estimates the model’s **average marginal response** to variation in a single skill dimension, marginalising over the observed distribution of all other job attributes.

### **Individual Conditional Expectation (ICE)**
ICE analysis extends PDP by retaining **job-level prediction trajectories** rather than averaging them.  
Using a random subsample of jobs, predictions were generated for each job across the same grid of values, holding all non-target features fixed at their observed values.

To improve interpretability, ICE curves were **centred relative to a baseline skill value** (the grid value closest to zero for each PCA), expressing results as changes in predicted salary. This removes dominant baseline salary differences across jobs and isolates the marginal effect of skill variation.

Categorical PDPs and ICE curves were intentionally excluded. For high-cardinality categorical variables, counterfactual averaging produces unrealistic feature combinations and offers limited additional insight. Categorical effects are instead examined through **SHAP attribution** and **residual-based fairness diagnostics**.

---

## **Results**

### **Average Effects (PDP)**
Across all evaluated skill components, PDP curves exhibit **piecewise-constant behaviour with discrete jumps and plateau regions**, consistent with the split-based structure of tree ensemble models such as XGBoost. The observed PDP shapes align closely with the dependence patterns identified in SHAP plots, indicating that the average marginal responses reflect the same underlying model structure.

- **Skill_PC1** shows a pronounced penalty at low values followed by rapid recovery and saturation, consistent with a baseline technical gatekeeping effect.
- **Skill_PC2** displays an overall downward shift in predicted salary across much of its range, reflecting movement toward lower-paying role regimes.
- **Skill_PC3** exhibits stepwise increases in predicted salary, indicating threshold-based recognition of modelling and ML depth.
- **Skill_PC8** remains largely flat across most of its empirical range, with changes appearing only near the edges of support, suggesting limited average marginal influence.

### **Heterogeneity and Stability (ICE)**
Centered ICE curves show **limited divergence and largely parallel trajectories** across jobs for all evaluated components. This indicates minimal interaction-driven heterogeneity and confirms that the PDP curves provide a faithful summary of the model’s global behaviour. Where deviations occur, they are concentrated near the extremes of the empirical support, reflecting sparsity rather than systematic instability.

---

## **Conclusions (PDP & ICE)**

PDP and ICE analyses together demonstrate that skill-related effects in the Salary Response Model operate primarily through **threshold and saturation dynamics**, with limited job-specific heterogeneity once structural attributes are accounted for. PDPs clarify the *average form* of these effects, while ICE validates their **stability across job contexts**.

Taken together with SHAP, this explainability suite provides a coherent and internally consistent account of salary formation in the model, grounding downstream analyses—such as skill value ranking, competitiveness scoring, and recommendation design—in transparent, well-validated model behaviour.



# **1.8 Global Skill Value Index (Model-Implied Skill Ranking)**

## **Overview**
To complement salary prediction, fairness diagnostics, and explainability, Chapter 1 produces a **Global Skill Value Index**: a descriptive ranking of individual skills based on how strongly they are associated with higher predicted salaries **within the fitted Salary Response Model**. This component is explicitly non-causal and is intended as an interpretability artefact rather than a downstream modelling dependency.

---

## **Methodology**
Skills enter the Salary Response Model through a latent representation: the 27 binary skill indicators are compressed into **10 PCA components** (`skill_PC1` … `skill_PC10`). To recover an interpretable skill-level signal, we combine:

1. **PCA loadings**, which quantify how strongly each original skill contributes to each skill component, and  
2. **SHAP-derived component influence**, which quantifies the average direction and magnitude with which each PCA component contributes to salary predictions.

For each PCA component, we compute a signed component weight from SHAP values (mean signed contribution combined with mean absolute magnitude). Skill-level values are then obtained by **back-projecting** these component weights through the PCA loading matrix and summing contributions across all components. Finally, the resulting skill scores are standardised into a z-scored index for interpretability.

The final output is exported as `skill_value_index.csv` and stored in `data/processed/`.

---

## **Results**
The resulting index provides a coherent global ranking across the 27 aggregated skill groups. Skills associated with specialised technical depth and modelling capability receive positive scores, while skills that primarily characterise lower-paying role regimes or non-specialised requirements receive neutral to negative scores. Importantly, the ranking reflects the model’s learned structure after controlling for job title, location, sector, company attributes, and the full latent skill composition.

Because the signal is computed from PCA components, the index captures **aggregate skill influence across multiple latent axes** rather than attributing salary effects to isolated binary indicators. This yields a stable, interpretable summary of how the salary model values skills in the dataset.

---

## **Conclusions**
The Global Skill Value Index provides a defensible, model-consistent interpretation of skill value in the labour market represented by the dataset. It should be interpreted as a descriptive summary of the salary model’s learned valuation logic, not as evidence of causal returns to specific skills. The index is included as an interpretability artefact for Chapter 1 and may be referenced in later chapters, but it is not required for downstream pipelines.

---

# Chapter 1 — Closure & Scope Boundary

## Chapter 1 Synthesis

Chapter 1 establishes the **mechanical core** of the Job Intelligence Engine.  
Building on the structured data foundation from Chapter 0, this chapter models how salary and skill requirements emerge from the interaction between job attributes, company characteristics, location, and latent skill structure.

The chapter delivers four primary analytical components:
1. A **Salary Response Model** that predicts expected salary conditional on job and company attributes.  
2. A **Skill PCA representation** that compresses sparse, correlated skill indicators into a stable latent space.  
3. A set of **Skill Requirement Models** producing a continuous job × skill probability matrix.  
4. A comprehensive **interpretability layer**, including residual-based fairness diagnostics and model explainability (SHAP, PDP, ICE).

In addition, Chapter 1 includes a **Global Skill Value Index** as an interpretive artefact summarising how individual skills are associated with predicted salary within the fitted model. This index is explicitly descriptive and is not a mechanical dependency for downstream chapters.

## Explicit Scope Boundary

Chapter 1 is intentionally limited to **system mechanics and diagnostics**.  
It does **not** perform individual optimisation, career guidance, or recommendation logic.  
Context-specific skill valuation (e.g. by city, sector, or role), fairness-adjusted skill recommendations, and decision-oriented trade-offs are **out of scope** at this stage and are deferred to later chapters where user positioning and objectives are defined.

With this boundary enforced, Chapter 1 provides a stable, interpretable, and reusable foundation for all subsequent modelling work in the Job Intelligence Engine.

---

# Chapter 1 → Chapter 2 Data Contract

This section defines the **guaranteed outputs** from Chapter 1 that downstream chapters are allowed to consume.  
Only the artefacts listed below are considered stable dependencies.

---

## Guaranteed Artefacts

1. Processed Job Dataset  
2. Salary Response Model 
3. Skill PCA Transformer
4. Skill Requirement Models
5. Skill Probability Matrix
6. Salary Residuals 

**Chapter 1 is now closed.**


# Chapter 2 — Hidden Structure (Graphs, Embeddings & Skill Ecosystems)

## Overview

Chapter 2 transforms the Job Intelligence Engine from feature-based modelling into **explicit structural discovery**.  
Rather than analysing how individual predictors influence outcomes (Chapter 1), this chapter asks how jobs and skills are organised *relative to one another* across the labour market.

Using the outputs of Chapter 1—most importantly the job × skill probability matrix—Chapter 2 constructs a relational representation of the market as a graph. From this graph, continuous embeddings are learned, interpretable job families are inferred, and skill ecosystems and specialisation patterns are extracted. The result is a reusable structural layer that supports ecosystem analysis, aggregation, and downstream recommendation logic.

---

## 2.1 Job–Skill Bipartite Graph

### Methodology

The labour market is represented as a **weighted bipartite graph** with two node types:

- **Job nodes**, one per `job_id`
- **Skill nodes**, one per skill group in the project taxonomy

Edges connect jobs to skills, with edge weights equal to the model-estimated probability that a given skill is required for a job (from Chapter 1’s skill requirement models). A probability threshold is applied to remove negligible connections while preserving continuous signal.

This approach replaces sparse binary indicators with a smooth, denoised representation of skill demand. Stable identifiers (`job_id`, skill names) are preserved throughout to ensure all graph-derived outputs can be joined back to the canonical modelling dataset.

### Results

The bipartite graph explicitly encodes shared skill neighbourhoods between jobs and shared job contexts between skills. Jobs are no longer independent rows but positions within a network of skill relationships.

### Conclusions

The job–skill graph establishes the structural backbone of Chapter 2. It converts tabular outputs into a relational object suitable for embedding methods and ecosystem-level analysis.

---

## 2.2 Node2Vec Embeddings (Jobs and Skills)

### Methodology

Node2Vec is applied to the weighted bipartite graph to learn low-dimensional embeddings for both job and skill nodes. Random walks sample local and higher-order graph structure, and a skip-gram objective maps nodes with similar connectivity patterns to nearby positions in vector space.

Embeddings are exported as versioned artefacts to ensure reproducibility and controlled iteration.

### Results

The resulting embeddings form a **continuous geometric representation** of the labour market:

- Jobs requiring similar constellations of skills cluster naturally in embedding space.
- Skills that co-occur across similar job contexts are embedded nearby.

These vectors replace high-dimensional co-occurrence patterns with compact representations suitable for clustering, similarity search, and ecosystem analysis.

### Conclusions

Node2Vec embeddings provide a noise-tolerant, reusable latent space capturing relational similarity across jobs and skills. They form the foundation for all subsequent structural inference in Chapter 2.

---

## 2.3 Job Family Discovery (Clustering Job Embeddings)

### Methodology

To convert embedding geometry into an interpretable structural artefact, job embeddings are clustered to infer **latent job families**. Prior to clustering, embeddings are L2-normalised so that Euclidean distance reflects directional similarity rather than magnitude.

KMeans is used to guarantee full assignment coverage—every `job_id` receives exactly one `job_family_id`. The number of clusters is selected using the silhouette score as a stability heuristic. The silhouette curve exhibits a broad plateau rather than a sharp optimum, indicating continuous structure rather than discrete separability. A value of *k = 20* is selected as a pragmatic balance between resolution and interpretability.

### Results

Each job is assigned to a single job family, producing a stable mapping (`job_id → job_family_id`) that can be joined to all downstream datasets. These families represent **job ecosystems**—groups of roles that share similar skill contexts, regardless of title noise or sector labels.

### Conclusions

Job family clustering produces the core interpretable output of Chapter 2. These families are unsupervised and provisional by design: they are intended as a structural navigation layer rather than a definitive taxonomy.

---

## 2.4 Skill Embeddings and Skill Ecosystems

### Methodology

While job embeddings are clustered, skill embeddings are treated differently. Skills are inherently overlapping and relational, so they are retained in continuous space rather than forced into hard clusters.

Skill embeddings are used to:
- Construct **skill similarity graphs** based on nearest-neighbour relationships
- Identify bundles of skills that tend to co-occur across jobs
- Highlight potential gateway or bridging skills that connect otherwise distinct job ecosystems

### Results

Skill similarity structures reveal coherent bundles (e.g. analytics + BI, ML + pipelines) and identify skills that act as connectors across multiple job families.

### Conclusions

Retaining skills in continuous embedding space preserves the richness of skill relationships and enables ecosystem-level reasoning that would be lost under hard clustering.

---

## 2.5 Skill Specialisation Maps (Lift-Based Analysis)

### Methodology

To interpret how skills are distributed across the market, **skill specialisation maps** are constructed. For each categorical variable (job family, sector, title, seniority, ownership, state, company size), the mean skill probability within each group is computed and compared to the global mean.

Specialisation is defined as **lift**:

\[
\text{skill\_lift}_{g,s} = \bar{P}(s \mid g) - \bar{P}(s)
\]

Positive values indicate over-representation, negative values under-representation, and near-zero values generic usage.

Heatmaps are used to visualise these patterns consistently across groupings.

### Results

Skill specialisation maps reveal:

- **Sectoral structure**: strong differentiation in ML, pipelines, and cloud skills across industries.
- **Title-driven signal**: job titles provide the clearest and strongest skill specialisation patterns.
- **Seniority gradients**: advanced skills and leadership skills increase systematically with seniority.
- **Ownership and size effects**: weaker but interpretable secondary patterns.
- **Geographic variation**: modest signal relative to sector and title, suggesting location is not a primary driver of skill composition.

### Conclusions

Skill specialisation maps provide the primary interpretability layer for Chapter 2. They connect abstract embeddings back to human-interpretable labour-market structure and confirm that sector and job type dominate skill demand patterns.

---

## 2.6 Chapter 2 Outputs and Role in the System

By the end of Chapter 2, the Job Intelligence Engine has produced:

- A probability-weighted **job–skill bipartite graph**
- **Node2Vec embeddings** for jobs and skills
- A stable **job family mapping** (`job_id → job_family_id`)
- **Skill similarity networks**
- **Skill specialisation maps** across key categorical dimensions

These artefacts define a reusable **structural layer** that bridges feature-based modelling (Chapter 1) and downstream applications. Rather than imposing premature taxonomies or optimisation objectives, Chapter 2 focuses on revealing how the market is organised.

This structural foundation supports future modules including:
- Ecosystem-aware recommendation systems
- Skill pathway and upskilling analysis
- Industry and domain specialisation summaries
- Competitiveness and transition modelling

The emphasis throughout Chapter 2 is on **discovering structure, not enforcing labels**—providing a faithful, data-driven map of how jobs and skills relate within the labour market.

# Chapter 3 — Individual Positioning  
## User-Centric Ranking, Competitiveness & Skill Gaps  
Date: 2025-12-18

---

## Overview

Chapter 3 introduces the **individual-facing analytical layer** of the Job Intelligence Engine.  
Where Chapters 0–2 establish data foundations, predictive mechanics, and latent market structure, Chapter 3 answers a fundamentally different question:

**How does the labour market look from the perspective of a specific individual?**

This chapter operationalises all upstream artefacts—skill representations, PCA space, salary structure, skill-demand probabilities, and job-family structure—into a deterministic framework for **user-level positioning**. The outputs are not abstract summaries but concrete, interpretable diagnostics:

- ranked job lists,
- calibrated skill gaps,
- competitiveness scores,
- and robustness checks on modelling assumptions.

Importantly, Chapter 3 does **not** optimise outcomes or prescribe decisions.  
It provides a transparent, explainable representation of *fit* and *difficulty*, leaving decision-making to downstream systems or human interpretation.

---

## Conceptual Framing

Chapter 3 decomposes individual positioning into two orthogonal dimensions:

1. **Suitability** — *How well does a job align with the user’s current profile and preferences?*  
2. **Competitiveness** — *How difficult would it be for the user to realistically access that job?*

These dimensions are intentionally separated. A job can be highly suitable but extremely competitive, or accessible but poorly aligned. Collapsing these into a single score would obscure meaningful trade-offs.

---

## 3.1 User Profile Schema (Single Entry Point)

### Methodology

Chapter 3 introduces a formal **UserProfile schema**, implemented in `schemas.py`, which acts as the **sole entry point** for individual-level analysis. Raw user inputs may include:

- free-text skill descriptions,
- optional structured skill flags,
- location preferences,
- sector and title constraints,
- salary targets,
- weighting preferences.

The schema performs the following steps deterministically:

1. **Validation and normalisation** of raw inputs.  
2. **Skill extraction** using the same taxonomy and extractor as Chapter 0.  
3. **Construction of the canonical 27-skill binary vector**.  
4. **Projection into the shared PCA skill space** using the stored PCA transformer from Chapter 1.  
5. Assembly of a fixed-shape, model-ready profile object.

### Results

The resulting UserProfile is guaranteed to be compatible with all Chapter 3 modules and directly comparable to job representations produced earlier in the pipeline.

### Conclusions

Centralising user parsing ensures:
- consistency across analyses,
- no duplication of preprocessing logic,
- and a strict separation between **data ingestion** and **analytical reasoning**.

---

## 3.2 Candidate Set Construction (Hard Constraints)

### Methodology

Before any scoring occurs, the system constructs a **candidate job set** by applying hard filters in a fixed, deterministic order:

1. State (location)  
2. Sector  
3. Enriched job title  
4. Job family  

Filtering is intentionally strict. If the resulting candidate set is empty, the pipeline raises an explicit error rather than degrading silently.

### Results

The candidate set represents the **feasible market slice** for the user under their stated constraints.

### Conclusions

This step enforces conceptual clarity: suitability and competitiveness are evaluated **only among jobs the user is genuinely willing to consider**.

---

## 3.3 Suitability Modelling

### Methodology

Suitability measures *alignment*, not difficulty or prestige. It is constructed from two components:

#### 1. Skill Match
Skill match is computed as **cosine similarity** between the user’s PCA skill vector and each job’s PCA skill vector:

\[
\text{skill\_match} \in [-1, 1]
\]

This value is linearly mapped to \([0, 1]\) for aggregation purposes, preserving relative ordering.

#### 2. Salary Alignment
Salary alignment is formulated as a **one-sided score**:
- Jobs meeting or exceeding the user’s target salary receive full credit.
- Jobs below target are penalised proportionally.
- Over-paying jobs are not penalised.

This reflects realistic preference structure: exceeding expectations does not reduce suitability.

#### Aggregation
The final suitability score is a weighted sum of normalised components. Weights are explicit, user-configurable, and automatically normalised to sum to one.

### Results

Each candidate job receives:
- raw and normalised skill match,
- salary alignment score,
- overall suitability score.

### Conclusions

Suitability answers the question:
**“If access were free, how good of a fit is this job for the user?”**

---

## 3.4 Skill Gap Analysis (Probability-Based)

### Methodology

To explain suitability rankings and support actionable insight, Chapter 3 computes **skill gaps** using the job × skill probability matrix from Chapter 1.

Procedure:
1. Select the top-*K* most suitable jobs.  
2. For each skill, compute the mean predicted probability across these jobs.  
3. For skills the user lacks, define gap severity as this mean probability.  
4. For skills the user already possesses, gap severity is zero.

This produces a ranked gap table where magnitude reflects **expected importance**, not binary absence.

### Results

The output is a calibrated skill-gap profile highlighting which skills most strongly limit access to the user’s preferred jobs.

### Conclusions

This approach avoids brittle binary logic and produces **probabilistic, market-informed upskilling signals**.

---

## 3.5 Competitiveness Modelling

### Methodology

Competitiveness measures **barrier-to-entry**, independent of desirability. It combines three components:

#### 1. Expected Missing Skill Burden
For each job, the probability-weighted sum of skills the user lacks is computed:

\[
\mathbb{E}[\text{missing skills}] = \sum_{s \notin \text{user}} P(s \mid \text{job})
\]

#### 2. Skill Rarity Weighting
Not all skills are equally substitutable. Missing a rare skill is more costly than missing a common one.  
Expected missingness is therefore weighted by **inverse global skill frequency**, computed from the full probability matrix.

#### 3. Salary Percentile
Jobs demanding compensation far above market norms are intrinsically more competitive. Each job’s salary is mapped to its percentile within the candidate set.

#### Aggregation
The components are combined into a single competitiveness index using explicit, documented weights.

### Results

Each job receives:
- expected missing skill burden,
- rarity-adjusted burden,
- salary percentile,
- final competitiveness score.

### Conclusions

Competitiveness answers:
**“How hard would it be for this user to access this job, regardless of fit?”**

---

## 3.6 Sensitivity Analysis (Robustness Diagnostics)

### Methodology

Suitability and competitiveness rely on explicit weighting assumptions.  
To test robustness, Chapter 3 performs **systematic sensitivity analyses**:

- Component weights are varied across a grid.
- Rankings under each configuration are compared to the baseline using **Spearman rank correlation**.
- Stability curves summarise how sensitive rankings are to modelling choices.

### Results

Sensitivity outputs identify:
- regimes of stable rankings,
- components that disproportionately influence outcomes,
- and configurations where rankings become unstable.

### Conclusions

This step does not optimise weights.  
It ensures that conclusions drawn from rankings are **robust rather than artefacts of arbitrary parameter choices**.

---

## 3.7 Outputs

Chapter 3 produces the following artefacts:

- Ranked candidate job list with suitability and competitiveness scores  
- Skill gap table with calibrated severity values  
- Sensitivity-analysis summaries for both suitability and competitiveness  

All outputs are deterministic, reproducible, and fully explainable.

---

## Chapter 3 Synthesis

Chapter 3 completes the transition from **market modelling** to **individual positioning**.  
It does so without introducing opaque optimisation or recommendation heuristics.

Key achievements:
- A single, rigorous user-entry schema  
- Explicit separation of fit and difficulty  
- Probabilistic, market-informed skill gap diagnostics  
- Robustness checks on all composite metrics  

This chapter establishes a principled foundation for future work, including:
- individual salary prediction,
- personalised recommendations,
- and dynamic career-path simulation.

Chapter 3 is intentionally diagnostic, not prescriptive.  
It tells the user **where they stand**, not **what they must do**.

---
