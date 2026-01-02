# Chapter 3 — UserProfile Schema (Block A)

## Purpose
Define a deterministic, minimal, and realistic user input contract for **individual positioning** in the Job Intelligence Engine.

This schema specifies:
- what information a user provides,
- how it is interpreted,
- and what guarantees downstream Chapter 3 modules can rely on.

---

## 1. Required Inputs

### 1.1 Skill Text
**Field:** `skill_text`  
**Type:** `str`

Free-form text describing the user’s skills, experience, or background  
(e.g. resume excerpt, LinkedIn summary, or manual list).

This text is processed using the **same normalization and dictionary-based
skill extraction pipeline as Chapter 0**, producing a canonical skill vector.

---

### 1.2 Location Context
**Field:** `current_state`  
**Type:** `str`

User’s current state, selected from the canonical state list used in the dataset.

Special value:
- `"ALL"` → no geographic restriction (national baseline)

This field defines the **baseline labour market context** for positioning.

---

## 2. Intent / Preference Inputs (Optional)

### 2.1 Target Job Title (choose one)

The user may specify **one** of the following:

#### a) Job Title Family
**Field:** `job_title_family`  
**Type:** `str`  
**Options:** 4 high-level title families

Used for coarse positioning and exploration.

#### b) Enriched Job Title
**Field:** `job_title_rich`  
**Type:** `str`  
**Options:** ~25 enriched title categories

Used for precise positioning.  
If both title fields are supplied, `job_title_rich` takes precedence.

---

### 2.2 Target Sector
**Field:** `target_sectors`  
**Type:** `list[str] | None`

List of sectors selected from the canonical sector vocabulary.

Sector acts as a **hard filter** on candidate jobs.

---

### 2.3 Salary Target
**Field:** `salary_target`  
**Type:** `float | int | None`

Desired or expected annual salary.

Used only for **salary alignment** in suitability scoring.
If `None`, salary alignment is skipped or neutral.

---

## 3. Derived Internal Representations

The following fields are **not user inputs**, but are deterministically derived:

- `skill_tokens`: list of normalized tokens extracted from `skill_text`
- `skills_by_group`: `{skill_group: 0/1}` over the 27 canonical skill groups
- `skill_vector`: ordered binary vector aligned to:
  - skill PCA transformer
  - salary model
  - skill probability matrix
- `unmapped_tokens`: tokens not matched to any skill group (for transparency/debugging)

---

## 4. Skill Handling Rules (Frozen)

- Multiple tokens mapping to the same skill group → group set to `1`
- Unmatched tokens:
  - dropped
  - recorded in `unmapped_tokens`
- Missing skill groups → treated as `0`
- Empty `skill_text` is valid (all-zero skill vector)

No skill proficiency, duration, or experience level is inferred.

---

## 5. Explicit Non-Goals

The following are intentionally excluded from Chapter 3:

- Skill proficiency levels
- Education or credentials
- Years of experience
- Company size, ownership, or seniority as user inputs
- User embeddings

These may be considered only in later chapters if justified.

---

## 6. Stability Guarantee

Once frozen:
- All Chapter 3 modules must accept this schema
- Any change requires explicit scope review
- Tests assume this exact behaviour

This schema is the **sole definition of an individual** for positioning.
