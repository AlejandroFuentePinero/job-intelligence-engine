# Job Intelligence Engine — Project Overview  
Narrative summary of Chapter 0 and the Salary Response Model of Chapter 1  
Date: 2025-12-09

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

# Chapter 1 — Salary Response Model (Narrative)

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

# How the Chapters Connect

- **Chapter 0** establishes a unified and structured representation of each job posting.  
- **Chapter 1** begins modelling the underlying mechanics of that structured world by predicting salary.  

Subsequent chapters will build on this foundation to model skill requirements, identify under- and over-paying market segments, learn job embeddings, quantify career paths, and develop personalised job recommendations.

This overview captures the conceptual story behind all work completed so far, without technical details or implementation specifics.
