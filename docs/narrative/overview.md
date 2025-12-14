# Job Intelligence Engine — Project Overview  
Narrative summary of Chapter 0 and the Salary Response Model of Chapter 1  
Date: 2025-12-14

The Job Intelligence Engine converts unstructured job postings into a structured analytical system.  
Its goal is to understand how job titles, skills, industries, and companies shape the labour market, beginning with salary prediction.  
This overview describes the conceptual purpose of Chapter 0 and the first modelling component of Chapter 1.

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


---

# How the Chapters Connect

- **Chapter 0** establishes a unified and structured representation of each job posting.  
- **Chapter 1** begins modelling the underlying mechanics of that structured world by predicting salary.  

Subsequent chapters will build on this foundation to model skill requirements, identify under- and over-paying market segments, learn job embeddings, quantify career paths, and develop personalised job recommendations.

This overview captures the conceptual story behind all work completed so far, without technical details or implementation specifics.
