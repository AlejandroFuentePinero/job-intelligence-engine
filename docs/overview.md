# Job Intelligence Engine — Chapter 0 Overview  
**Scope: Data Acquisition, Standardisation & Feature Foundation**

---

## 1. Purpose of Chapter 0
Chapter 0 establishes the **clean, standardised, modelling-ready dataset** that the entire Job Intelligence Engine depends on.  
Its role is to transform heterogeneous job postings into a **single unified dataset** with consistent structure, interpretable features, and high-quality signals for downstream modelling (salary prediction, skill modelling, career pathways, job embeddings, etc.).

This chapter is the *data backbone* of the project: it ensures every analysis, model, and insight in later chapters is anchored to clean, consistent, and reproducible data engineering.

---

## 2. High-Level Workflow
Chapter 0 takes two raw datasets — Data Analyst roles and Data Scientist roles — and performs a multi-step engineering pipeline:

1. **Load & Align Raw Sources**  
   - Read `da_jobs_raw.csv` & `ds_jobs_raw.csv`  
   - Align mismatched columns  
   - Introduce a `role_source` tag  
   - Standardise missing values (`-1` → NA)

2. **Core Feature Cleaning**  
   - Extract U.S. state of role and headquarters  
   - Normalise ownership type (`public`, `private`, `nonprofit`, etc.)  
   - Convert `Founded` to numeric year  
   - Remove unusable, high-missing, or irrelevant metadata  
   - Harmonise international listings

3. **Text Processing & Enrichment**  
   - Clean job descriptions (stopword removal, punctuation normalisation)  
   - Extract seniority using both title and description  
   - Produce a clean `job_title_base` plus hierarchical `job_title_family`  
   - Build a unified text pool (`title_plus_description`) for skill extraction

4. **Domain Assignment**  
   - Map job titles to a curated taxonomy (DA, DS, ML Scientist, etc.)  
   - Assign known domain where possible using a lookup table

5. **Salary Parsing & Standardisation**  
   - Convert raw salary strings (annual/hourly) into  
     `sal_min`, `sal_max`, `sal_mean`  
   - Drop intermediate parsing artifacts

6. **Skill Extraction (Dictionary-Based v1)**  
   - Use the curated taxonomy of 1300+ skill tokens  
   - Extract binary features for each  
     **(domain × skill_level)** bucket  
   - Produce a complete multi-hot skill matrix

7. **Missing Data Strategy**  
   - Drop rows with **no salary AND no description**  
   - Fill missing `Industry`, `Sector`, `Size` with `"Unknown"`  
   - Keep missing `Rating` and `Founded` for modelling flexibility

8. **Pruning & Saving Final Output**  
   - Drop noisy, redundant, or intermediate engineering columns  
   - Save the final **47-column, 6162-row** modelling-ready dataset  
     as `chapter0_processed_jobs.csv`

---

## 3. Outputs of Chapter 0

### ✔ **Primary Output**  
`chapter0_processed_jobs.csv`  
- 6162 rows  
- 47 fully curated features  
- No structural inconsistencies  
- Consistent encoding of text, numeric, categorical and multi-hot skill features  
- Guaranteed safe for downstream modelling without re-cleaning

### ✔ **Supporting Output**  
`chapter0_domain_lookup.csv`  
- Domain mapping table used during title-based domain assignment  
- Provides transparency and reproducibility for domain decisions

---

## 4. What Chapter 0 Guarantees

### **4.1 Consistent Feature Space Across All Roles**
Regardless of variation in job titles, salary formats or text formatting, every role emerges with:
- A standardised family-level job title  
- Domain assignment  
- Fully parsed salary estimates  
- Multi-hot skill profile  
- Cleaned description usable for advanced extraction (e.g., ML/NLP later)  
- Strong categorical and company metadata  

### **4.2 Deterministic, Reproducible Pipeline**
All transformations are deterministic Python functions — no notebook ad-hoc steps.  
Running the pipeline again will always produce **bit-identical** output.

### **4.3 Clear Downstream Readiness**
Chapter 0 explicitly prepares the data for Chapters 1–5:
- Salary modelling  
- Skill embeddings (SBERT v2, graph features)  
- Job–job similarity  
- Career pathways  
- Skill gap modelling  
- Multi-objective optimisation for upskilling routes  

---

## 5. Why Chapter 0 Matters  
Most job intelligence projects fail at the foundation: messy text, missing structure, inconsistent titles, fragmented skills, and unreliable salary signals.

Chapter 0 solves all of this.

It provides:
- A **clean schema**  
- A **structured feature space**  
- A **unified job representation**  
- The **standardisation required for modelling**  

This is the chapter that makes the rest of the project *possible*.

---

## 6. Summary
Chapter 0 is complete.  
It delivers a rigorous, transparent, and production-quality dataset ready for modelling.

From this point forward, the project shifts from **data engineering** → **analysis & modelling**.

This file documents everything required to understand and reproduce the foundations of the Job Intelligence Engine.

---
