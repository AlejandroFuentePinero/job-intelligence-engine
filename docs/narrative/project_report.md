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



---

# Current Status & Next Steps

## Completed
- Full Chapter 0 data foundation  
- PCA-based skill dimensionality reduction  
- Salary Response Model v4  
- Evaluation and benchmark tools  

## Next Steps (Chapter 1)
- Per-skill logistic requirement models  
- Salary fairness residual analysis  
- SHAP, PDP, and ICE explainability suite  

---

# Summary
Chapters 0 and 1 lay the foundational analytical structures of the Job Intelligence Engine, including a robust data pipeline, structured feature definitions, and the first predictive models. These components enable the subsequent modelling of skill demand, job similarities, labour-market competitiveness, and personalised career recommendations.
