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

# Chapter 3 — Individual Positioning

## Overview

Chapter 3 introduces the first **user-facing analytical layer** of the Job Intelligence Engine.  
Building on the structural and probabilistic artefacts produced in Chapters 0–2, this chapter reframes the system from a market-level analysis into an **individual positioning framework**.

Rather than predicting outcomes or prescribing actions, Chapter 3 addresses a more fundamental question:

Given a user’s current skill profile and constraints, how does the labour market position them relative to available jobs?

To answer this, the chapter formalises three complementary concepts: **suitability**, **competitiveness**, and **skill gaps**.

---

## Core Contributions

### Suitability Scoring (Opportunity Alignment)

Suitability quantifies how well a job aligns with the user’s current profile.

It combines two signals:
- Skill alignment, measured as similarity between the user’s skills and job skill requirements in shared latent space  
- Salary alignment, measured as whether the job meets or exceeds the user’s stated salary target  

Suitability is explicitly user-centric:
- Higher values indicate better alignment  
- There is no penalty for exceeding salary expectations  
- Scores are normalised to support ranking rather than absolute interpretation  

Suitability answers the question:  
**How well does this job fit the user right now?**

---

### Competitiveness Index (Barrier to Entry)

Competitiveness captures how difficult a job is for the user to access, independent of how attractive it may be.

It integrates two orthogonal barriers:
- Expected missing skills, computed as the probability-weighted burden of skills the user lacks  
- Salary percentile, reflecting how ambitious the job is relative to the candidate set  

Skill missingness is computed using the job–skill probability matrix rather than binary indicators.  
Rare skills contribute more strongly to the barrier via inverse-frequency weighting, ensuring that missing scarce skills are penalised more than missing ubiquitous ones.

The final competitiveness index is normalised to a [0, 1] scale:
- Higher values indicate greater barriers  
- Competitiveness is a diagnostic signal, not an optimisation target  

Competitiveness answers the question:  
**How hard would this job be to obtain for the user?**

---

### Skill Gap Analysis (Actionable Deficits)

Skill gap analysis identifies which missing skills most constrain the user’s access to their best-matching opportunities.

The analysis:
- Focuses on the user’s top-ranked jobs by suitability  
- Computes the average probability that each skill is required across those jobs  
- Assigns gap severity only to skills the user currently lacks  

This produces a calibrated ranking of missing skills grounded in real job demand rather than raw frequency or generic curricula.

Skill gaps answer the question:  
**Which missing skills matter most for the jobs the user is already close to?**

---

### Sensitivity Analysis (Robustness Diagnostics)

Chapter 3 includes sensitivity analyses to assess the stability of results under reasonable changes in weighting assumptions.

Two dimensions are evaluated:
- Suitability weighting between skill alignment and salary alignment  
- Competitiveness weighting between skill barriers and salary barriers  

Rank stability is assessed using rank-correlation metrics against baseline configurations.  
Results show that rankings remain stable across plausible parameter ranges, indicating that conclusions are not driven by arbitrary weighting choices.

---

## System Integration

All Chapter 3 logic is orchestrated through a single positioning pipeline that:
- Loads validated artefacts from previous chapters  
- Constructs a structured user profile  
- Filters candidate jobs deterministically  
- Computes suitability, competitiveness, and skill gaps  
- Optionally evaluates robustness through sensitivity analysis  

No component in Chapter 3 retrains models, mutates upstream artefacts, or duplicates preprocessing logic.  
All outputs are deterministic and reproducible.

---

## Scope Boundary

Chapter 3 is intentionally descriptive and diagnostic rather than prescriptive.

It does not:
- Predict individual salaries  
- Optimise career trajectories  
- Recommend specific actions or learning paths  
- Model time, effort, or transition costs  

All outputs are framed as **positioning signals**, not decisions.

---

## Role in the Job Intelligence Engine

Chapter 3 completes the transition from market-level structure to individual-level positioning.

It provides the analytical foundation required for subsequent chapters that will:
- Estimate user-specific salary potential  
- Simulate skill acquisition pathways  
- Support recommendation and optimisation logic  
- Integrate agentic decision-making systems  

---

## Chapter 3 Closure

Chapter 3 establishes a coherent, interpretable, and reusable framework for mapping an individual into the labour-market structure learned by the Job Intelligence Engine.

The chapter deliberately avoids premature optimisation, focusing instead on:
- clarity of signals  
- robustness of diagnostics  
- alignment with learned market structure  

With this chapter complete, the system is prepared to move from **positioning** to **decision support and recommendation**, which are addressed in subsequent chapters.

**Chapter 3 is now closed.**


# Chapter 4 — Recommender Engine (v1)

Date: 2025-12-23  
---

## Overview
Chapter 4 begins the transition from **diagnostic positioning** (Chapter 3) to **actionable recommendation outputs**. The chapter reuses Chapter 3 as the canonical source of user-aligned candidates and derived skill features, then layers a lightweight recommender interface that (i) filters by suitability, (ii) separates opportunities into accessibility buckets via competitiveness, (iii) attaches an individualized salary prediction signal based on the user’s skill profile, and (iv) returns an inspectable explanation layer for the Top-N results.

---

## 4.1 Chapter 4 Context Loader (Wrapper)
**Module:** `features/artefacts_ch4.py`  
**Function:** `load_ch4_context()`

### Methodology
- Calls Chapter 3 public API (`run_positioning`) to construct:
  - `profile` (validated UserProfile + derived `skill_pcs`)
  - `candidates_df` (hard-filtered job set with suitability + competitiveness + encoded categorical codes)
  - `gap_df` and optional sensitivity outputs
- Loads aligned Chapter 3 artefacts via `load_ch3_artefacts()`:
  - `jobs_df` (refined full jobs table)
  - `skill_prob_matrix` (job×skill probability matrix)
- Loads the persisted salary model (`salary_model_v4.pkl`)
- Constructs a salary design matrix for inference by:
  - selecting job/company categorical codes from `candidates_df`
  - broadcasting the user’s 1×10 PCA skill vector across all candidate rows
  - returning a single “ready-for-Chapter-4” context payload

### Outputs
Returns a context dict containing:
- `profile`, `candidates_df`, `gap_df`, `sensitivity_out`
- `jobs_df`, `skill_prob_matrix`
- `salary_model`
- `user_salary_model_features` (candidate-level salary feature matrix with user PCs)

### Conclusions
This wrapper centralises all Chapter 4 dependencies while avoiding expensive reruns of Chapter 1. It establishes a reproducible and aligned input contract for downstream recommenders.

---

## 4.2 Chapter 4 Entrypoint Evaluation (Engineering Hardening)
**File:** `evaluation/chapter_4_entrypoint_eval.py`

### Methodology
A focused evaluator validates the Chapter 4 entrypoint and salary-feature construction with checks covering:
- artefact integrity and job_id alignment (jobs df vs skill matrix)
- determinism / invariance under identical inputs
- salary feature broadcasting shape + prediction smoke tests
- NA / schema sanity checks for core outputs

### Conclusions
The evaluation confirms that the Chapter 4 context loader is stable, aligned, and produces prediction-ready salary features without silent failure modes.

---

## 4.3 Hybrid Job Recommender (v1: Retrieve → Rerank; 2 Buckets)
**Module:** `features/job_recommender.py`  
**Function:** `job_recommender()`

### Methodology
1) **Load context:** calls `load_ch4_context()` to obtain candidates + salary model + salary features.  
2) **Attach salary predictions:** predicts `pred_sal` for candidate rows using the loaded salary model and user-broadcasted PCs.  
3) **Retrieve:** applies a suitability threshold (base, with optional floor fallback if the candidate set is too small).  
4) **Bucket:** splits jobs into two accessibility buckets using a competitiveness cutoff:
   - `best_now` (≤ `C_max`)
   - `stretch` (> `C_max`)
   Warnings are raised when bucket sizes fall below minimums, rather than forcing unsuitable recommendations.  
5) **Rerank:** computes a combined score `suitability - α * competitiveness_index` and sorts deterministically.  
6) **Summarise:** returns tables for candidate jobs and top-N recommendations per bucket, plus salary comparison summaries (expected vs predicted).

### Outputs
Returns a structured result dict with:
- `tables`: `candidate_jobs`, `top_best_now`, `top_stretch`
- `counts`: bucket sizes and candidate counts
- `salary_summary`: mean expected vs predicted salary and deltas
- `params`: key threshold and scoring parameters used

### Conclusions
Chapter 4 produces a shortlist of jobs organised by accessibility (best-now vs stretch), with individualized salary prediction signals attached to each recommended role.

---

## 4.4 Job Explanations (v1: Inspectable Recommendations)
**Module:** `features/job_explanations.py`  
**Function:** `build_job_explanations(rec, tau=0.5, validate=True)`

### Methodology
- Consumes the `job_recommender()` payload (`rec`) and enriches the Top-N recommendation tables.
- Adds deterministic rationale fields:
  - `why_bucket`: bucket assignment rationale using `competitiveness_index` and `c_max`
  - `why_rank`: explicit score decomposition using `suitability`, `competitiveness_index`, and `alpha`
  - `salary_context`: interpretable comparison of `sal_mean` (market expected) vs `pred_sal` (user-conditioned), including the gap
- Attaches Chapter 3 alignment diagnostics per job:
  - merges `skill_match_norm` and `expected_missing_norm` from `rec["tables"]["candidate_jobs"]` by `job_id`
- Adds interpretable skill-family coverage and missingness per recommended job:
  - thresholds `skill_prob_matrix` at `tau` to infer required families
  - compares against `profile["derived"]["skill_vector"]` to compute:
    - `missing_families`, `covered_families`
    - `n_missing_families`, `n_covered_families`
- Includes lightweight validation (optional) to fail loudly if contracts break (keys/columns, numeric coercions, merged metrics).

### Outputs
Returns a dict with:
- `tables`: `top_best_explained`, `top_stretch_explained`
- `metric_glossary`: column-level definitions for interpretation and reporting
- `meta`: `tau` used for requirement thresholding

### Conclusions
The explanation layer makes Chapter 4 recommendations auditable and user-facing without adding heavy downstream complexity. It provides transparent ranking logic, salary context, and interpretable skill-family coverage for each recommended role.

---

## 4.5 Upskilling Recommender (v1: Counterfactual Skill Gains on a Frozen Universe)
**Module:** `features/upskilling_recommender.py`  
**Function:** `upskill_recommender(...)`

### Methodology
- Runs the baseline `job_recommender()` to obtain the constrained `scored_universe`, then **freezes** the job universe via `candidate_override_df` (job_id-only) so every upskill scenario is directly comparable.
- Uses `build_job_explanations()` on the frozen universe to retrieve per-job `missing_families` (thresholded at `tau`), focusing on **stretch** jobs to define the candidate upskill families.
- For each missing family, injects representative tokens from that family (derived from job descriptions using the same extractor dictionary) into `skill_text`, reruns `job_recommender()` on the frozen universe, and computes **per-job deltas** vs baseline (percentage-point deltas for bounded 0–1 metrics).
- Summarises each upskill scenario using a composite impact score that rewards:
  - **promotion_rate** (baseline stretch → best_now),
  - **mean score gains** (especially among baseline-stretch jobs),
  and penalises:
  - **demotions** (baseline best_now → stretch) and worst-tail harms in best_now (10th percentile), with an explicit demotion guardrail (`demotion_tol`).

### Outputs
Returns a dict with:
- `job_base_upskill`: long-table (job_id × scenario) with baseline + per-scenario deltas and `bucket_movement`
- `upskill_summary`: scenario-level metrics (promotion/demotion rates, score deltas, tail risk) + composite `upskill_impact_score`
- `upskill_recommendation`: Top-N ranked skill families for upskilling
- `missing_dict`: raw family → matched tokens (from stretch-job descriptions)
- `recommendation_dict`: deduped example tokens per recommended family for reporting

### Conclusions
The upskilling module provides a quantitative “what to learn next” layer by simulating skill acquisition as counterfactual profile changes over a fixed job market slice. This produces stable, auditable upskilling recommendations that integrate cleanly into Chapter 5 reporting (top families + example tokens + measurable positioning gains and promotion rates).

---

## 4.6 Career Simulation (v2 / optional: User-Defined What-If Scenarios)
**Module:** `features/career_simulation.py`  
**Function:** `career_simulation(...)`

### Methodology
- Runs a baseline `job_recommender()` call to obtain the constrained `scored_universe`, then **freezes** the job universe via `candidate_override_df` (`job_id`-only) so every scenario is directly comparable.
- Accepts explicit user-defined scenarios (e.g., “add SQL + dbt”, “add AWS + Docker”) and, for each scenario:
  - injects a small, deduplicated set of tokens into the user `skill_text`,
  - reruns `job_recommender()` on the **same frozen universe**, and
  - computes **per-job deltas** vs baseline (percentage-point deltas for bounded 0–1 metrics).
- Produces both job-level and scenario-level diagnostics:
  - `bucket_movement` (promoted / demoted / unchanged) based on `best_now` vs `stretch`,
  - promotion and demotion rates,
  - mean delta score on baseline-stretch jobs (upside) and baseline-best jobs (risk),
  - a tail-risk proxy using the 10th percentile delta score among baseline-best jobs.
- Includes explicit **no-op detection** aligned to v1’s curated dictionary constraint:
  - scenarios are skipped when injected tokens do not change the extracted user `skill_vector` (prevents misleading “effects” from out-of-vocabulary tokens).
- Supports a scenario-level guardrail (`demotion_tol`) to reject scenarios that materially harm baseline `best_now` opportunities.

### Outputs
Returns a dict with:
- `baseline_universe`: baseline scored universe (one row per `job_id`)
- `scenario_jobs`: long-table (job_id × scenario) containing scenario metrics
- `deltas`: scenario rows merged to baseline metrics, with:
  - `delta_skill_match_norm`, `delta_suitability`, `delta_expected_missing_norm`,
  - `delta_competitiveness_index`, `delta_score`,
  - `bucket_movement`
- `scenario_summary`: scenario-level impact table (promotion/demotion rates, mean deltas, tail risk, `passes_guardrail`)
- `top_unlocked_jobs`: top-*N* promoted jobs per scenario (stretch → best_now), ranked by `delta_score`
- `scenario_meta`: scenario audit log (kept/skipped, reasons, tokens used)

### Conclusions
Career simulation exposes the same frozen-universe counterfactual machinery used by upskilling, but with user-defined scenarios instead of auto-derived missing families. It is deferred to v2 because its reliability depends on broader skill-token coverage and more robust skill normalisation; without that, many user-proposed tokens can become no-ops, reducing interpretability and portfolio value relative to the v1 upskilling module.

---


## Current Scope Boundary
Chapter 4 v1 intentionally prioritises a minimal, working recommender loop with inspectable outputs. The following remain out of scope for the current closure point:
- upskilling recommender and skill ROI-style indices
- what-if career simulation and cross-state optimisation
- orchestrator pipeline and full end-to-end recommender test suite
- persistence of Chapter 4 artefacts for dashboards and reproducibility

Chapter 4 is in active development, with a stable entrypoint, a functioning v1 job recommender, and an explanation layer now established.


# Assumptions & Limitations

This section documents the modelling assumptions and practical limitations at two levels:
1) **Project-level** (cross-cutting constraints), and  
2) **Chapter-level** (module-specific constraints that shape interpretation and scope).

---

## Overall Project Assumptions

- **Static labour-market snapshot**: the dataset represents a single slice of postings (no reliable time series or forecasting).
- **Postings approximate requirements**: job descriptions are treated as meaningful signals for skills, seniority, and role structure.
- **Taxonomy can normalize reality**: noisy titles and heterogeneous descriptions can be mapped into a stable job-title family space.
- **Skill extraction is “good enough”**: token-based parsing captures the dominant skill signals needed for modelling and recommendation.
- **Models capture associations, not causation**: downstream recommendations interpret model response surfaces, not causal effects.
- **Geography/sector labels are informative**: location and sector categories are assumed sufficiently consistent for aggregation and comparison.

## Overall Project Limitations

- **No supply-side competitiveness**: no applicant volumes, interview difficulty, or hiring funnel outcomes—competitiveness is a proxy.
- **No temporal dynamics**: cannot detect changing demand, seasonality, or emerging skills; results can become stale as the dataset ages.
- **Data source bias**: postings overrepresent certain industries, company types, and job styles; conclusions may not generalize globally.
- **Partial observability**: many real drivers (team quality, visa sponsorship, remote eligibility, brand, negotiation, portfolio strength) are not encoded.
- **Limited user representation**: user profiles are primarily skill-token based (no proficiency, recency, depth, or evidence unless explicitly provided).
- **Interpretability boundaries**: explanations support transparency, but they are not guaranteed to mirror employer reasoning.

---

## Chapter 0 — Foundations (Preprocessing & Taxonomy)

### Assumptions
- Raw postings contain enough consistent structure (title, description, location, sector, salary fields) to be normalized.
- Title normalization can reliably collapse noisy variants into a usable job-family taxonomy.
- Dictionary/regex-based skill parsing provides a usable first-order skill signal.

### Limitations
- Missing/incorrect fields (salary ranges, sectors, locations) can propagate noise downstream.
- Title normalization can misclassify edge cases (hybrid roles, ambiguous titles, non-standard seniority terms).
- Skill extraction is lossy (synonyms, implicit requirements, context-dependent meaning) and can be biased by boilerplate text.

---

## Chapter 1 — System Mechanics (Salary, Skill Demand, Fairness)

### Assumptions
- Salary is modelable as a function of title family, skills, geography, and company/job traits (association-based).
- Skill-requirement models approximate “probability a posting mentions/needs a skill” using observable features.
- Residual-based fairness views are informative as *relative* over/underpay signals given the modeled covariates.

### Limitations
- Compensation is noisy and context-dependent (negotiation, equity, leveling, remote premiums, bonuses) and not fully captured.
- Skill requirement probabilities depend on text mention and extraction quality (not true necessity).
- Fairness/residual analyses are descriptive; omitted-variable bias can make “over/underpay” conclusions non-causal.
- Calibration and uncertainty are limited unless explicitly implemented (e.g., quantiles, prediction intervals).

---

## Chapter 2 — Hidden Structure (Graphs, Clusters, Ecosystems)

### Assumptions
- Job–skill co-occurrence is a meaningful proxy for latent similarity between jobs and between skills.
- Node2Vec embeddings preserve useful neighbourhood structure for downstream clustering and similarity queries.
- KMeans job-family clustering yields interpretable “ecosystems” for macro navigation and recommendations.

### Limitations
- Embeddings/clusters depend on hyperparameters and graph construction choices; different settings can change structure.
- Co-occurrence can reflect posting style, not true functional skill relationships.
- Clusters are not ground-truth ontologies; boundaries are fuzzy and may merge distinct subdomains.
- Similarity edges can overconnect generic skills (e.g., “communication”) unless explicitly handled.

---

## Chapter 3 — Individual Positioning (Suitability, Competitiveness, Gaps)

### Assumptions
- A user can be positioned against jobs using match signals derived from skill overlap, embeddings, and constraints.
- Competitiveness can be proxied using rarity/requirements/seniority signals (not applicant supply).
- Skill gaps inferred from posting requirements are actionable targets for upskilling guidance.

### Limitations
- User proficiency, years of experience, and evidence strength are not fully modeled unless explicitly provided.
- Suitability is a model score, not a hiring probability; “fit” includes many unobserved factors.
- Competitiveness is a proxy and can disagree with real difficulty in the market.
- Gap analysis inherits extraction noise and posting idiosyncrasies; missing skills may be overstated for some postings.

---

## Chapter 4 — Recommender Engine (Recommendations, Upskilling, Simulation)

### Assumptions
- **Skill text is a usable proxy for capability** and maps reliably to the internal skill vector representation.
- **Frozen-universe counterfactuals are comparable**: upskilling/simulation reruns on the same `job_id` set so deltas are interpretable.
- **Ranking weights reflect preferences**: composite scores (skill vs salary, promotion/demotion tradeoffs) encode user objectives.

### Limitations
- No supply-side signal (applicant pool, interview bar); recommendations optimize model fit, not guaranteed outcomes.
- No time dimension; recommended skills/jobs may shift as the market evolves.
- Counterfactual upskilling is not causal: it measures **model response** to added skills, not real learning ROI or hiring uplift.
- Explainability is partial: explanations surface missing families/matches, not full human reasoning.
- One-step skill injection: does not model learning time, prerequisites, curricula, or multi-skill sequencing unless extended.

---

## Chapter 5 — Insights & Dashboards (Transparency & Exploration)

### Assumptions
- Aggregated views (salary landscapes, skill demand, job-family structure) provide useful macro-level decision support.
- Persisted artefacts (models, embeddings, cluster centroids, summary tables) are stable enough to back interactive exploration.

### Limitations
- Dashboards are descriptive; they can mislead if interpreted causally or used outside dataset scope.
- Aggregation can hide heterogeneity (within-sector variance, company effects, niche subfamilies).
- Without temporal data, dashboards cannot answer “what’s changing” questions reliably.
- Interpretability is bounded by upstream modelling and extraction quality; dashboards reflect those constraints.

---
# Job Intelligence Engine — Project Report (Working Draft, Completeness-First)
Dates covered: 2025-12-11 → 2025-12-30

This report is intentionally “wide” and descriptive. Its purpose is to ensure everything built so far is documented with enough detail that it can later be tightened into:
- a professional technical report,
- a pseudo-academic paper-style writeup,
- and a reproducible project README / documentation bundle.

Where numbers appear, they must be backed by the project’s evaluation outputs. If any metric values are currently placeholders, label them as “VERIFY FROM EVAL OUTPUTS”.

---

# A) Project-level Overview (What the system is)

The Job Intelligence Engine is a modular, deterministic pipeline that turns a static snapshot of Glassdoor job postings into:

1) **Clean, validated job records** with stable categorical encodings and engineered features (Chapter 0).
2) **Predictive + probabilistic modelling layers**:
   - a salary model (continuous regression),
   - and 27 skill-family probability models (binary classifiers producing calibrated probabilities),
   plus interpretability and diagnostics (Chapter 1).
3) **A structural market layer**:
   - a weighted job–skill bipartite graph,
   - embeddings for jobs and skills,
   - job families via clustering,
   - skill similarity networks,
   - and specialisation (“lift”) maps across groupings (Chapter 2).
4) **User-centric positioning**:
   - constrained candidate selection,
   - suitability and competitiveness scoring,
   - probabilistic skill gaps,
   - and sensitivity analyses (Chapter 3).
5) **Decision support / recommender layer**:
   - best-now vs stretch buckets,
   - user-conditioned salary prediction attached to candidates,
   - auditable explanations,
   - and counterfactual upskilling (v1),
   with optional scenario simulation (v2/experimental) (Chapter 4).

Design priorities throughout:
- Determinism and reproducibility.
- Explicit data contracts between chapters.
- Auditable outputs (tables + invariants + evaluators).
- Clear separation between “diagnostics” (positioning) and “prescription” (recommendations).

---

# B) Chapter 0 — Data Acquisition, Cleaning & Feature Foundation

## B.1 Inputs and dataset scope
- Input source: Glassdoor job postings (static snapshot; not a time series).
- Role focus: data-centric roles (e.g., DS/DA families), unified into a single canonical dataset.
- Key constraint: postings are treated as “signals of requirements”, not ground-truth skill necessity.

## B.2 Cleaning and standardisation (deterministic preprocessing)
**Primary objective:** produce a coherent, joinable, model-ready schema with stable categorical fields and consistent text fields.

Core transformations (deterministic):
- Schema alignment and merge across raw sources.
- Missing-value normalisation and placeholder removal.
- Location normalisation to state-level signals.
- Company metadata normalisation (sector, ownership, size, etc.).
- Salary text parsing into numeric features (and preserving missing salary as a valid state).
- Text cleaning for descriptions (to standardise the extraction surface).

Outputs include:
- canonical job identifier (`job_id`) and a stable row-level join key for downstream artefacts.

## B.3 Title processing + seniority extraction + domain assignment
**Motivation:** raw titles are noisy and non-standard; many downstream components require a stable title representation.

Outputs:
- multiple title-normalisation fields (`job_title_raw`, `job_title_base`, `job_title_norm`, `job_title_family`)
- seniority signals extracted from title + description, combined into a single seniority field

Domain assignment:
- a precomputed title embedding + clustering mapping (SBERT-based title embeddings clustered into semantic “domain” groupings),
  used as a deterministic lookup layer for title → domain.

## B.4 Salary parsing & normalisation
Core outputs:
- `sal_min`, `sal_max`, `sal_mean`, `sal_is_hourly`

Key design choice:
- salary parsing is treated as a preprocessing responsibility; all modelling assumes salary is already numeric and standardised.

## B.5 Skill extraction (dictionary-based v1; canonical 27-family space)
- Skills extracted from combined text fields (title + cleaned description).
- A curated token dictionary maps mentions to 27 aggregated skill families.
- Output is a consistent 27-dimensional multi-hot representation per job.

Important limitation (explicit):
- dictionary-based extraction is lossy and sensitive to synonyms and phrasing.
- later counterfactual modules (upskilling/simulation) inherit this constraint (out-of-vocabulary tokens can become no-ops).

## B.6 Chapter 0 artefact outputs (canonical)
- Processed jobs table (canonical modelling input for Chapter 1 pipelines).
- Benchmarks / sanity checks (Chapter 0 validator exists as an evaluation module).

---

# C) Chapter 1 — System Mechanics (Models, Diagnostics, Interpretability)

## C.1 Objective
Chapter 1 builds the “mechanical inference layer”:
- salary prediction conditioned on job + company + latent skills
- probabilistic skill-demand inference across 27 skill families
- interpretability and diagnostics: fairness/residuals, SHAP, PDP/ICE, and derived interpretive artefacts

## C.2 Feature engineering for modelling
- Categoricals are encoded into stable integer codes suitable for model ingestion.
- Title signals are enriched (domain + family combined into a “richer” categorical signal).
- Skill PCA components (`skill_PC1..skill_PC10`) are appended for model stability and reduced collinearity.

## C.3 Skill PCA (27 → 10 latent axes)
Purpose:
- reduce collinearity and compress the skill space into orthogonal latent dimensions.
- ensure consistent inference for both job rows and user profiles.

Artefact:
- persisted PCA transformer (`skill_pca_v1.pkl`)
- job-level PCA features in the modelling dataset (`skill_PC1..skill_PC10`)

Validation principle (project-level):
- PCA correctness is validated by reproducible downstream behaviour (salary model metrics match the exploratory workflow when the same pipeline is used).

## C.4 Salary Response Model (XGBoost v4)
Objective:
- predict expected salary (`sal_mean`) from:
  - categorical job/company attributes
  - enriched title representation
  - PCA skill components

Evaluation:
- standard regression metrics (R², RMSE, MAE)
- residual diagnostics
- feature importance and explainability

Note on numbers:
- any specific metric values should be sourced from `evaluation/salary_model_eval.py` outputs and/or saved evaluation tables.

Artefact:
- persisted salary model (`salary_model_v4.pkl`)
- salary predictor utility for inference usage (model module).

## C.5 Skill requirement models (27 binary classifiers → probability matrix)
Core concept:
- train 27 independent models, one per skill family, to estimate:
  P(skill_family_k required | job attributes)

Key output:
- a dense job × skill probability matrix (continuous 0–1), replacing brittle binary indicators for downstream analysis.

Why probabilities matter (downstream impact):
- enables calibrated skill-gap analysis (Chapter 3),
- supports graph edge weights (Chapter 2),
- enables smoother similarity and specialisation analysis.

Evaluation:
- ROC/PR AUC, calibration diagnostics, feature importances.
- IMPORTANT: keep the results descriptive unless backed by the persisted evaluation outputs.

Artefacts:
- 27 persisted skill-family models
- job × skill probability matrix builder + exported matrix.

## C.6 Fairness / residual diagnostics (model-adjusted pay patterns)
Method:
- residual = observed − predicted salary
- group residuals by key categorical dimensions:
  state, sector, title family, company size, ownership, seniority

Interpretation boundary:
- descriptive, not causal (omitted variables exist; “over/underpay” is relative to the model’s covariates).

Outputs:
- per-dimension residual summary tables (CSV) + plots.

## C.7 Explainability suite (SHAP + PDP/ICE)
Goal:
- explain model valuation logic (SHAP)
- validate response shapes and heterogeneity (PDP/ICE)

Key interpretability deliverables:
- global SHAP importance across all predictors
- component-level SHAP dependence views for PCA axes
- PDP/ICE to confirm thresholding/saturation patterns and limited heterogeneity

Engineering note:
- plots should be saved deterministically; file overwrite behaviour should be verified so that blank/partial outputs cannot persist silently.

## C.8 Global skill value index (interpretive artefact)
Purpose:
- translate latent PCA/SHAP signals into a stable, human-readable skill-family ranking.

Boundary:
- descriptive; not causal; not required for downstream mechanics.

Artefact:
- `skill_value_index.csv` (if produced and persisted).

---

# D) Chapter 1 → Chapter 2 Data Contract (Guaranteed dependencies)
Downstream chapters may rely only on the following Chapter 1 outputs as stable dependencies:

1) Processed jobs table with stable IDs and categorical codes
2) Persisted PCA transformer (`skill_pca_v1.pkl`)
3) Job-level PCA features (`skill_PC1..skill_PC10`) or the deterministic procedure to generate them
4) Persisted salary model (`salary_model_v4.pkl`) and its required feature schema
5) Persisted 27 skill-family models OR the exported job × skill probability matrix
6) Exported job × skill probability matrix (preferred dependency for Chapter 2 and Chapter 3)
7) Evaluation tables/plots used as documentation evidence (not mechanical dependencies)

Chapter 1 is “closed” only if these artefacts are persisted and reproducible end-to-end.

---

# E) Chapter 2 — Hidden Structure (Graphs, Embeddings, Job Families, Skill Ecosystems)

## E.1 Objective
Convert the probabilistic skill-demand layer into an explicit representation of labour-market structure:
- graph representation (jobs ↔ skill families, weighted)
- shared embedding space for jobs and skills
- job families inferred from job embeddings
- skill ecosystems inferred from skill similarity
- specialisation maps (“lift”) across groupings

## E.2 Weighted job–skill bipartite graph
Construction:
- node types: `job_id` nodes and 27 skill-family nodes
- edge weights: skill probability values from Chapter 1 probability matrix
- thresholding: low-probability edges can be removed to reduce noise while preserving graded weights for retained edges

Outputs:
- persisted graph artefact (gpickle or equivalent) for reuse in embedding training and validation.

Core invariant:
- all graph-derived outputs must remain joinable back to `job_id` and canonical skill-family names.

## E.3 Node2Vec embeddings (jobs and skills)
Training:
- Node2Vec run on the bipartite graph to produce vector embeddings for both jobs and skills in a shared space.
- hyperparameters are configurable; metadata should be saved with artefacts to preserve reproducibility.

Outputs:
- job embeddings table keyed by `job_id`
- skill embeddings table keyed by canonical skill-family names

Optional diagnostic (recommended to document explicitly):
- lightweight embedding stability check (e.g., nearest-neighbour overlap) to ensure embeddings are not degenerate.

## E.4 Job families via clustering job embeddings
Procedure:
- normalise embeddings (L2) prior to clustering so distance reflects angular similarity.
- cluster with KMeans to guarantee full assignment coverage (each job gets exactly one family).

Output:
- a stable mapping `job_id → job_family_id` usable in downstream aggregation and recommendation.

Interpretation boundary:
- “job families” are structural clusters, not ground-truth ontologies; boundaries are fuzzy.

## E.5 Skill ecosystems (similarity structure over skill embeddings)
Procedure:
- compute similarity between skill embeddings (cosine similarity via dot products after normalisation).
- retain top-k neighbours per skill to form a sparse skill–skill graph / edge list.

Output:
- skill similarity edge list (`skill_1`, `skill_2`, `similarity`) used for ecosystem interpretation and (later) co-learning suggestions.

## E.6 Skill specialisation maps (“lift” by grouping variables)
Goal:
- interpret how skill probabilities deviate from global baselines within groups:
  job families, sector, title family, seniority, ownership, state, company size.

Output:
- for each grouping variable, a table capturing lift values across 27 skill families,
  suitable for heatmaps and reporting.

## E.7 Chapter 2 outputs summary (reusable structural layer)
- job–skill bipartite graph
- job embeddings + skill embeddings
- job families mapping
- skill similarity networks
- skill specialisation maps (lift tables)

---

# F) Chapter 2 → Chapter 3 Data Contract (Guaranteed dependencies)
Chapter 3 may rely on:

1) Modelling-ready jobs table (with stable IDs and categorical encodings)
2) Job-level PCA features or deterministic PCA transform procedure
3) Job × skill probability matrix (preferred; enables calibrated gaps and competitiveness)
4) Job family mapping (optional for aggregation and reporting; not required for core positioning)
5) Skill similarity edges (optional; used later for macro suggestions / co-learning neighbour skills)

---

# G) Chapter 3 — Individual Positioning (User → Ranked Jobs, Skill Gaps, Competitiveness)

## G.1 Objective
Provide a deterministic, explainable framework for placing an individual into the labour market slice defined by their constraints.

Core outputs:
- candidate job set
- suitability scores (fit)
- competitiveness scores (barrier)
- probabilistic skill gaps
- robustness checks via sensitivity analysis

## G.2 Single entrypoint: UserProfile schema
Responsibilities:
- validate and normalise user inputs (skills text + constraints)
- extract skills into the same canonical 27-family representation used for jobs
- project the user into the shared PCA skill space
- produce a fixed-shape, deterministic profile object consumed everywhere else

Key invariant:
- downstream modules must not re-implement user parsing; they consume the UserProfile output.

## G.3 Candidate selection (hard constraints; controlled failure)
Hard filters applied in a fixed order (deterministic).
Explicit failure mode:
- if filters yield an empty set, raise an error rather than silently returning nonsense.

## G.4 Suitability (fit)
- skill match: similarity in PCA space between user and job
- salary alignment: one-sided “meet/exceed target” scoring
- weighted combination with explicit weights (auto-normalised)

## G.5 Competitiveness (barrier to entry)
- expected missing skill burden: probability-weighted missingness
- rarity weighting: missing rare skills penalised more
- salary percentile: ambition proxy within candidate set
- combined into a competitiveness index

Interpretation boundary:
- competitiveness is a proxy (no applicant supply-side data).

## G.6 Probabilistic skill gaps (actionable deficits)
Computed from top-K suitable jobs:
- mean required probability per skill family across those jobs
- gap severity only for skills the user lacks

## G.7 Sensitivity analysis (robustness)
- vary weights across a grid
- compare ranking stability vs baseline (Spearman correlation)
- output stability diagnostics for suitability and competitiveness.

## G.8 Chapter 3 outputs
- `candidates_df` with suitability/competitiveness columns
- `gap_df` with ranked skill gaps
- sensitivity tables (when enabled)

Important report hygiene:
- This report currently contains two Chapter 3 writeups. Keep one canonical version and merge any unique details into it.

---

# H) Chapter 3 → Chapter 4 Data Contract (Guaranteed dependencies)
Chapter 4 relies on:

1) Chapter 3 public API output: `profile`, `candidates_df`, `gap_df`, sensitivity outputs
2) Chapter 3 artefact loader output: `jobs_df`, `skill_prob_matrix`
3) Persisted salary model artefact and its required feature schema
4) Stable categorical code columns inside `candidates_df` (required for salary inference features)
5) A stable join key (`job_id`) preserved everywhere

---

# I) Chapter 4 — Recommender Engine (v1)

## I.1 Objective
Move from “diagnostics” to “actionable decision support” while remaining auditable and deterministic:
- produce recommendations as two buckets: best-now vs stretch
- attach user-conditioned salary predictions to candidates
- provide explanation strings and interpretable skill-family missingness
- provide a counterfactual upskilling layer on a frozen job universe (v1)
- optionally support user-defined scenario simulation (v2/experimental)

## I.2 Context loader (Chapter 4 wrapper)
`load_ch4_context()` centralises dependencies:
- calls Chapter 3 public API to get candidate universe + derived user PCs
- loads `jobs_df` and `skill_prob_matrix`
- loads the persisted salary model
- builds salary inference features by broadcasting user PCs across candidate rows + selecting required categorical codes

Critical invariant:
- feature matrix rows must align 1:1 with `candidates_df` rows (no silent merges).

## I.3 Hybrid recommender (retrieve → rerank; 2 buckets)
Core mechanics:
1) attach user-conditioned salary predictions to each candidate job
2) apply suitability gating with fallback floors if candidate set is too small
3) bucket jobs by competitiveness threshold into best-now and stretch
4) rerank within each bucket by a composite score:
   score = suitability − alpha * competitiveness_index
5) deterministic tie-breaking (stable sorts) so results are reproducible

Outputs (structured dict):
- scored universe table
- top-best-now table
- top-stretch table
- parameters, counts, salary summaries, warnings

## I.4 Explanation layer (auditable Top-N)
Adds:
- why_bucket, why_rank, salary_context
- missing_families vs covered_families using thresholded `skill_prob_matrix`
- metric glossary for reporting

Includes internal validation (fail loudly when contracts break).

## I.5 Upskilling recommender (v1; frozen-universe counterfactuals)
Key design:
- freeze the job universe (job_id-only override) so all scenarios are comparable.
- derive candidate missing skill families from stretch jobs.
- extract representative tokens for those families from job descriptions.
- inject tokens into the user skill text and rerun recommender on the same frozen universe.
- compute deltas and rank skill families by positioning gains while enforcing demotion guardrails.

Outputs:
- long scenario table (job_id × scenario)
- scenario summaries (promotion/demotion rates, score deltas, tail risk)
- top recommended families + example tokens for reporting.

## I.6 Career simulation (v2/experimental; user-defined scenarios)
Same frozen-universe machinery, but scenarios are specified by the user.
Includes “no-op detection” when added tokens don’t change the extracted skill vector (dictionary coverage limit).

## I.7 Chapter 4 evaluation and integration hardening
Two evaluation levels exist conceptually:
1) entrypoint eval: validate context loader + salary feature broadcasting + artefact alignment.
2) pipeline eval: validate orchestration contracts across recommender → explanations → upskilling → simulation (when enabled),
   including frozen-universe invariants and output-schema checks.

IMPORTANT:
- The report’s “Current Scope Boundary” must match what is actually implemented in `src/` now.
  If upskilling is implemented and used, it is not “out of scope” anymore; mark simulation as optional/experimental if that is accurate.

---

# J) Cross-cutting engineering notes (should exist in the final report)

## J.1 Determinism and reproducibility
- Stable identifiers (`job_id`) preserved across all chapters.
- Artefacts are persisted as files (models, matrices, embeddings) and loaded via dedicated loader modules.
- Evaluators exist to prevent silent contract drift (schema checks, invariance checks, smoke tests).

## J.2 Artefact versioning (recommended to document explicitly)
For each persisted artefact:
- filename convention (with version tags where relevant)
- producing pipeline/module
- consumed-by modules downstream
- minimal schema contract (expected columns / keys)

## J.3 Separation of concerns
- features: pure transformations and scoring utilities
- models: training/inference modules
- evaluation: diagnostics and contract validation
- pipelines: orchestration scripts for reproducible builds

---

# K) Assumptions & limitations (retain, but add two missing “big ones”)

Add explicitly:
1) **Dictionary coverage constraint** impacts upskilling/simulation:
   counterfactuals measure model response only when injected tokens map to known skill families.
2) **Relative scoring constraint**:
   suitability/competitiveness are defined relative to the candidate universe after hard filters; changing constraints changes the “market slice” and therefore changes interpretation.

(Keep the existing assumptions/limitations section; it is broadly correct.)

# Additions to increase report completeness (paper + technical report ready later)
Date: 2025-12-30

This block is designed to be pasted into the project report as new sections (or appended to relevant chapters). It adds the missing “paper-critical” and “ship-critical” content: (A) paper-style framing + contributions, (B) unified methods spine, (C) results anchors for Chapters 2–4, (D) evaluation narrative, and (E) reproducibility / usage notes.

---

# 0. Paper-Style Framing (Problem, Contributions, Research Questions)

## Motivation / Problem Statement
Modern data roles (Data Analyst, Data Scientist, ML Engineer, Data Engineer) are defined less by titles and more by *skill bundles* and *market context* (sector, location, company type). However, job postings are noisy: titles are inconsistent, skill requirements are expressed in heterogeneous language, and salary information is partially missing or non-standardised. This makes it difficult to (i) measure skill demand reliably, (ii) compare jobs in a structured way, and (iii) position an individual against the market in a transparent, auditable manner.

The Job Intelligence Engine addresses this by converting raw postings into a deterministic, multi-layer representation of the labour market:
1) a cleaned and validated job dataset,
2) predictive models for salary and skill-demand probabilities,
3) a relational job–skill graph and embedding space capturing latent structure,
4) an individual positioning layer (fit vs barrier-to-entry) grounded in probabilities rather than binary indicators, and
5) a recommendation layer with explanations and counterfactual upskilling simulations on a frozen candidate universe.

## Intended User and Use Cases
Primary user: an individual transitioning into data roles (or repositioning within them) who can provide a free-text skills description plus optional hard constraints (location, sector, role family, salary target).  
Primary use cases:
- understand what roles are “best-now” vs “stretch” given current skills and constraints,
- identify which missing skill families most constrain access to preferred roles,
- quantify the model-implied positioning gains from acquiring specific skill families (counterfactual simulation).

Secondary use cases (market analysis):
- describe salary landscapes and “over/underpay” patterns conditional on role and skill structure,
- identify job families and skill ecosystems from co-occurrence structure.

## Key Contributions (Portfolio/Paper Claims)
1. **Deterministic end-to-end pipeline** from raw postings to reusable artefacts (models, matrices, graphs, embeddings, clustering outputs), designed for reproducibility and inspection.
2. **Probabilistic skill-demand layer**: 27 skill-family models produce a dense job × skill probability matrix enabling calibrated gap analysis and structural market representations.
3. **Relational market representation**: a probability-weighted job–skill bipartite graph plus Node2Vec embeddings provide a latent space for job similarity and skill ecosystem structure.
4. **Individual positioning framework** that explicitly separates:
   - **Suitability** (alignment) from
   - **Competitiveness** (barrier-to-entry),
   with robustness checks via sensitivity analysis.
5. **Actionable recommender layer** that returns inspectable “best-now” and “stretch” buckets plus:
   - human-readable explanations, and
   - counterfactual upskilling recommendations under a strict frozen-universe invariant.

## Research Questions (Paper-Friendly)
RQ1: Can job postings be transformed into a stable, interpretable skill-demand representation suitable for downstream structural analysis and individual positioning?  
RQ2: Does a probability-based skill-demand matrix improve the interpretability and actionability of skill gaps compared to binary mention features?  
RQ3: Does a graph/embedding representation recover coherent job families and skill ecosystems that align with known labour-market role structure?  
RQ4: Can individual positioning be decomposed into fit vs barrier-to-entry in a way that is robust to weighting assumptions and produces stable ranked outputs?  
RQ5: Do counterfactual skill-injection simulations on a frozen job universe yield stable, auditable upskilling recommendations that translate into measurable positioning gains?

---

# 1. Unified Methods Spine (Cross-Chapter, Paper-Grade)

This section provides a single coherent methodological narrative. Chapters remain as implementation structure, but this “spine” is what a paper/report will be written from.

## 1.1 Data Source and Scope
Input data consist of raw Glassdoor job postings for data-related roles. The dataset is treated as a *static labour-market snapshot* (no time-series inference). Records are uniquely identified by a stable `job_id` and contain fields for title, description, location, sector/industry metadata, and (when present) salary text/ranges.

## 1.2 Deterministic Cleaning and Canonical Schema (Chapter 0)
All preprocessing is deterministic and produces a canonical processed dataset used by downstream pipelines. Key operations:
- schema alignment and merge of raw sources,
- missing-value normalisation (placeholder handling + explicit NA conventions),
- extraction of state-level location,
- standardisation of company metadata (sector, ownership, size, founding year),
- controlled text cleaning for descriptions (consistent casing/whitespace/boilerplate handling),
- removal or down-weighting of unreliable or sparse fields (explicitly documented in preprocessing code).

Output is a modelling-ready table with stable identifiers, curated categorical fields, cleaned description text, and derived salary features.

## 1.3 Title Normalisation and Domain Assignment (Chapter 0)
Job titles are normalised into multiple representations to reduce variance:
- raw title, base cleaned title, normalised title, and a job-family mapping (`job_title_family`).
Seniority signals are extracted from both title and description and combined into a single seniority feature.

Domain assignment uses a precomputed SBERT + KMeans title embedding/clustering artefact to map noisy titles into semantically coherent domains. This step increases interpretability and stabilises downstream modelling by compressing high-variance title strings into consistent categories.

## 1.4 Salary Parsing and Target Construction (Chapter 0)
Salary text is parsed into numeric ranges and harmonised to annual salary when necessary. Derived features include `sal_min`, `sal_max`, and `sal_mean` (midpoint). Hourly salaries are converted to annual equivalents using a consistent conversion rule (documented in code). Missing salary is preserved as “missingness” rather than being imputed into the target variable.

## 1.5 Skill Extraction and Taxonomy (Chapter 0)
A curated dictionary of skill tokens maps free text (title + description) into a canonical 27 skill-family space. Extraction is primarily token-based with multi-word matching support and case normalisation. The output is a multi-hot vector per job in the 27-family space. This representation is intentionally interpretable and deterministic, providing a stable foundation for both modelling and user profiling.

## 1.6 Skill Latent Space via PCA (Chapter 1)
Because the 27 skill-family indicators are correlated, PCA is fit to the full skill matrix and the top 10 components are retained as a compact latent representation (`skill_PC1..skill_PC10`). This reduces collinearity and provides a continuous embedding for skill composition used by the salary model and user-to-job similarity computations. PCA correctness is validated operationally by confirming that downstream model metrics reproduce the exploratory notebook results under the same pipeline.

## 1.7 Salary Response Model (Chapter 1)
Salary prediction models expected salary (`sal_mean`) from:
- categorical job/company attributes (state, sector, ownership, size, seniority, enriched title representation), and
- skill PCA components.
XGBoost is used with categorical handling and hyperparameter tuning (GridSearchCV). Evaluation uses standard regression metrics (R², RMSE, MAE) and residual diagnostics. The salary model is treated as *associational*, not causal.

## 1.8 Skill Requirement Models and Probability Matrix (Chapter 1)
To estimate skill demand more robustly than binary mentions alone, 27 binary classifiers are trained (one per skill family). Each model predicts a probability that the given skill family is required for a job, producing a dense job × skill probability matrix. Models are evaluated using ROC AUC, PR AUC, and calibration metrics (Brier score / calibration curves). The probability matrix is the primary downstream artefact: it enables calibrated skill gaps, graph construction, rarity weighting, and structural analyses.

## 1.9 Relational Market Representation: Graph + Embeddings (Chapter 2)
A weighted bipartite graph connects job nodes to skill-family nodes with edge weights derived from the skill probability matrix (optionally thresholded to remove negligible edges). Node2Vec is trained on this graph to learn embeddings for both jobs and skills in a shared latent space. These embeddings are persisted as reusable artefacts and form the basis for:
- job-family clustering (KMeans on job embeddings after L2 normalisation),
- skill ecosystems (k-nearest-neighbour similarity edges among skill embeddings),
- specialisation maps (lift-based aggregation of skill probabilities across groupings).

## 1.10 Individual Positioning: Suitability vs Competitiveness (Chapter 3)
Users enter through a single schema (`UserProfile`) that:
- validates inputs,
- extracts skills via the same taxonomy/extractor used for jobs,
- constructs a canonical 27-family user vector, and
- projects it into PCA space (`skill_PC1..skill_PC10`).

Candidate selection applies hard constraints in a deterministic order (state → sector → enriched title → job family). Scoring is then computed on this candidate set:
- **Suitability** combines skill-match (cosine similarity in PCA space) and salary alignment (one-sided target score).
- **Competitiveness** proxies barrier-to-entry via probability-weighted missing skills (with optional rarity weighting) and salary percentile within the candidate set.

Sensitivity analysis perturbs component weights across a grid and measures rank stability via Spearman correlation against baseline.

## 1.11 Chapter 4 Decision Support: Recommendations + Explanations + Counterfactual Upskilling
Chapter 4 uses Chapter 3 outputs as the canonical candidate universe and derived user features. It adds:
- user-conditioned salary prediction per candidate job via salary model inference on a constructed design matrix (categorical codes + broadcasted user PCs),
- a two-bucket recommender (best-now vs stretch) based on competitiveness thresholding,
- an explanation layer that makes ranking and bucket assignment auditable, and
- counterfactual skill-injection simulations on a **frozen job universe** to estimate positioning gains from learning skill families.

Frozen-universe invariance is a core methodological constraint: all scenarios compare the same job_id set, enabling meaningful deltas.

---

# 2. Results Anchors Needed for Chapters 2–4 (Add as “to-be-filled” placeholders)

This section makes later paper/report writing easy by explicitly stating what quantitative summaries and artefact references will be used. Where exact values are not currently recorded in this report, include a placeholder and link it to an artefact file produced by pipelines/evaluations.

## 2.1 Chapter 2 — Structural Outputs (Quantitative Anchors)
Add the following anchor bullets (fill with values later from saved artefacts):

- **Graph construction summary**
  - # job nodes: [FILL]
  - # skill nodes: 27
  - # edges after threshold τ_graph: [FILL]
  - mean/median degree (jobs): [FILL]
  - mean/median degree (skills): [FILL]
  - Artefact: `data/processed/job_skill_bipartite_*.gpickle`

- **Node2Vec training settings**
  - embedding dim: [FILL]
  - walk length / num walks: [FILL]
  - p/q: [FILL]
  - random seed: [FILL]
  - Artefacts: `data/processed/job_embeddings_node2vec_v01.csv`, `.../skill_embeddings_node2vec_v01.csv`

- **Embedding stability diagnostic (lightweight)**
  - nearest-neighbour overlap or similar sanity metric: [FILL]
  - Artefact: [FILL: diagnostic output path if stored]

- **Job-family clustering**
  - chosen k: 20 (current)
  - silhouette summary (plateau evidence): [FILL]
  - Artefact: `data/processed/job_families_graph_embeddings.csv` (+ optional silhouette plot if saved)

- **Skill ecosystem edges**
  - k-neighbours per skill: [FILL]
  - # edges retained: [FILL]
  - Artefact: `data/processed/skill_similarity_edges_k*_embeddings.csv`

- **Specialisation maps**
  - groupings produced (title_rich, job_family, sector, etc.)
  - Artefact: `data/processed/job_family_skill_specialisation.csv` (and related outputs)

These anchors convert Chapter 2 from “conceptual description” into “reportable results” with minimal later work.

## 2.2 Chapter 3 — Positioning Outputs (Quantitative Anchors)
Add anchor bullets for a standard demo run (using demo user config once Chapter 4 pipeline is finalised):

- candidate set size after hard filters: [FILL]
- distribution summaries:
  - suitability mean/median/IQR: [FILL]
  - competitiveness mean/median/IQR: [FILL]
- top-K gap profile:
  - top 5 missing skill families and their mean required probabilities: [FILL]
- sensitivity robustness summary:
  - suitability rank Spearman ρ range over weight grid: [FILL]
  - competitiveness rank Spearman ρ range over weight grid: [FILL]
- evaluator status:
  - chapter3_pipeline_eval: PASS/FAIL + key checks passed: [FILL]

## 2.3 Chapter 4 — Recommender Outputs (Quantitative Anchors)
For the same demo config:

- suitability gating:
  - s_min_base used: [FILL]
  - fallback triggered? (yes/no): [FILL]
  - eligible universe size: [FILL]
- buckets:
  - best_now size: [FILL]
  - stretch size: [FILL]
  - overlap check: 0 (must be 0) / status: [FILL]
- salary signal:
  - mean sal_mean vs mean pred_sal per bucket: [FILL]
  - mean predicted salary jump (stretch − best_now): [FILL]
- explanation layer:
  - example explanation fields present: PASS/FAIL [FILL]
- upskilling:
  - number of candidate missing families discovered from stretch jobs: [FILL]
  - top N recommended families: [FILL]
  - promotion_rate for best scenario: [FILL]
  - demotion_rate for best scenario: [FILL]
  - frozen-universe invariant check: PASS/FAIL [FILL]
- evaluator status:
  - chapter4_entrypoint_eval: PASS/FAIL [FILL]
  - chapter4_pipeline_eval: PASS/FAIL [FILL]

These anchors are what make later “Results” writing fast and credible.

---

# 3. Evaluation Narrative (Paper/Report Credibility Layer)

## Evaluation Philosophy
Evaluation is treated as two complementary layers:
1) **Model evaluation** (predictive performance and calibration) for Chapter 1 models, and
2) **Engineering evaluation** (determinism, invariants, contracts, boundary conditions) for Chapter 2–4 pipelines.

This project prioritises *reproducibility and auditable decision support* over purely maximising predictive metrics.

## 3.1 Chapter 1 Evaluation
- Salary model: R² / RMSE / MAE + residual diagnostics + explainability (SHAP, PDP/ICE).
- Skill models: ROC AUC + PR AUC + calibration metrics (Brier score, calibration curves), recognising prevalence-driven PR variability.

## 3.2 Chapter 2 Evaluation (Integrity over “accuracy”)
Because embeddings and clusters are unsupervised, evaluation focuses on integrity and stability:
- artefact integrity checks (IDs, shapes, missingness),
- sanity checks on graph density and degree distribution,
- stability checks (nearest-neighbour overlap or similar),
- clustering plausibility checks (silhouette plateau evidence, coverage).

## 3.3 Chapter 3 Evaluation (Behavioural and Robustness Checks)
- determinism and invariance under identical user input,
- controlled failure modes for empty candidate sets,
- score range and ranking invariants,
- sensitivity stability: rank correlations across weight grids.

## 3.4 Chapter 4 Evaluation (Orchestration Hardening)
- feature matrix integrity for salary inference (required columns, no NA, broadcast correctness),
- recommender output contracts (required tables, sorted ranks, disjoint buckets),
- explanation contracts (required fields, list columns consistent with counts),
- frozen-universe invariant across upskilling/simulation scenarios (critical),
- scenario audit logging (kept/skipped reasons).

---

# 4. Reproducibility & Shipping Notes (Technical Report Essentials)

## 4.1 Determinism and Seeds
All pipelines are designed to be deterministic given:
- fixed preprocessing rules,
- fixed model artefacts persisted to disk,
- fixed random seeds for stochastic components (e.g., Node2Vec),
- stable sorting tie-breakers (e.g., include `job_id` as last-order key).

Where stochasticity exists, it is controlled and documented.

## 4.2 Core Entrypoints
Public APIs:
- `run_positioning()` (Chapter 3): returns `profile`, `candidates_df`, `gap_df`, sensitivity outputs.
- Chapter 4 pipeline entrypoint (when finalised): runs recommender + optional explanations + optional upskilling/simulation using a persisted demo config.

## 4.3 Artefact Versioning Strategy
Persisted artefacts are versioned (e.g., `*_v01`, `*_v4`) to allow iteration without breaking downstream contracts. The report should treat only explicitly “guaranteed artefacts” as stable dependencies.

## 4.4 Demo Config as the Regression Anchor
A single `recommender_demo.json` is used as:
- the documentation-friendly “example run,” and
- the evaluation regression anchor (pipeline smoke test).
Optionally persist a small “golden snapshot” of the demo outputs (tables) to support future refactors.

---

# 5. Discussion Hooks (For Later Paper Writing)

This section is not final discussion text; it is a structured set of “hooks” that will later become a paper Discussion.

## 5.1 Why probability-based skill demand matters
Binary skill extraction is sparse and style-dependent. The probability matrix smooths this signal and enables calibrated gap analysis and network construction that is less sensitive to idiosyncratic posting language.

## 5.2 Structural vs individual layers
Chapters 1–2 describe the market; Chapter 3 reframes that structure from a user-centric viewpoint. The separation between suitability and competitiveness is essential for transparent trade-offs.

## 5.3 Counterfactual simulations as decision support (not causality)
Upskilling recommendations and career simulation measure *model response* under controlled skill injections on a frozen universe. They are useful for relative prioritisation within the system but do not guarantee labour-market returns.

## 5.4 Expected generalisation limits
The system is trained on a specific snapshot. Portability depends on distribution shift, evolving skill vocabularies, and changes in sectoral salary regimes.

---

# 6. “Stop: Move On” Note
With these sections added, the report contains:
- paper-grade framing,
- a unified methods spine,
- explicit results anchors for later filling from artefacts,
- a coherent evaluation narrative,
- shipping/reproducibility notes,
which is sufficient to later craft both a scientific-paper style narrative and a professional technical report.


## Chapter 5 — Insights & Dashboards (App) — Reproducibility Notes

Chapter 5 is a **product surface** (Streamlit app) plus a small set of **deterministic build assets** that make the app fast and reproducible. No model training occurs in Chapter 5: the app either (i) loads prebuilt, persisted artefacts for market context and explainability, or (ii) calls the **Chapter 4 recommender pipeline** on demand when the user clicks *Run recommender*.

### Inputs and required artefacts

**App runtime configuration**
- `src/job_intel/evaluation/recommender_demo.json`  
  Demo persona configuration used by the *Load demo persona* button (manual inputs remain supported).

**Chapter 5 persisted assets (fast app loading)**
- `data/processed/ch5_assets/fairness_group_summary_long.csv`  
  Group-level residual summaries used by the fairness explorer.
- `data/processed/ch5_assets/fairness_residual_box_stats.json`  
  Residual distribution summary statistics shown in the UI.
- `data/processed/ch5_assets/skill_value_index.csv`  
  Global Skill Value Index (GSVI) used by the skill value ranking view.
- `data/processed/ch5_assets/shap_salary_explanation.npz`  
  Salary-model SHAP explanation bundle (global bar + beeswarm + local categorical views).

**Upstream data consumed by Chapter 5 views**
- `data/processed/df_with_residuals.csv`  
  Residual series used for the fairness histogram and as input to the fairness asset builder (must include `residuals` column).
- `CH1_PROCESSED_SALARY_MODEL_PCA_DF` (path defined in config)  
  Chapter 1 salary-model PCA dataframe used for code→label lookup tables in local SHAP categorical plots (e.g., `state_code`→`state`, `sector_code`→`Sector`).
- `data/processed/skill_similarity_matrix/skill_similarity_edges_k5_embeddings.csv`  
  Chapter 2 k-NN edge list used by the macro co-learning visualisation (skill similarity around top recommended upskills).

### Build entrypoint (deterministic; no training)

Chapter 5 assets are built and validated via a single deterministic entrypoint:

- `python -m src.job_intel.pipelines.ch5_app_build`

This build step:
1) regenerates fairness artefacts into `data/processed/ch5_assets/` via `ch5_build_fairness_assets.py`  
2) validates that all required runtime artefacts for the app are present on disk (fail-fast if missing)

### App execution (product surface)

The Streamlit app is launched with:

- `streamlit run app.py`

The app is composed of modular pages (Home, Landscape, Recommender, Upskilling + Macro). The recommender and upskilling views call the **Chapter 4 pipeline** directly; Chapter 5 does not re-implement recommendation logic.

### Determinism, caps, and scope boundaries (v1)

- **No training in Chapter 5.** All artefacts are either deterministic build outputs (fairness assets) or precomputed outputs from Chapters 1–2.
- **Career simulation is disabled in v1.** Chapter 4 is executed with `run_career_sim=False` and no scenarios/config are applied.
- **Hard caps are enforced** in the UI for readability (e.g., top-N bars/categories/skills; top explained jobs per bucket). These caps are presentation constraints only and do not change the underlying pipeline outputs.
- **Interpretation is descriptive, not causal.** Landscape residuals and SHAP explainability describe learned patterns in the dataset and model behaviour; they do not establish causal effects.

### Key assumptions and limitations

- Salary drivers are dominated by **structural context** (location, role semantics, sector, company context); skills primarily refine outcomes within those regimes (skills appear as PCA bundles in the salary model).
- Skill similarity in the macro co-learning view reflects **embedding-space proximity** (skills that co-occur or are learned together in the job market), not a strict prerequisite ordering.
- Chapter 2 cluster/job-family artefacts are not exposed in the v1 UI to avoid unlabeled, high-cognitive-load views; Chapter 2 contributes to v1 primarily through the **skill similarity edges** used for co-learning.
