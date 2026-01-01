# Chapter 4.1 — Recommendation Intents + Scoring Objectives (v1)

## 1) Candidate Universe (Input Contract)
Chapter 4 operates ONLY on the Chapter 3 candidate universe (post-filtering + positioning):
- one row per `job_id`
- includes (at minimum): `suitability`, `competitiveness`, and salary context for display (e.g., salary percentile)
- includes enough metadata for display + role slicing (e.g., normalized title and/or job family)

Chapter 4 does not expand the candidate universe; it only ranks, explains, and simulates within it.

---

## 2) Supported Intents (v1)
### Intent A — Balanced (default)
Goal: recommend roles that maximize fit while minimizing barrier (salary shown for context, not primary).

### Intent C — Role-Focused
Goal: user targets a role slice (job family or title keyword) and we rank within that slice using the same Balanced logic.

We explicitly drop “High-Pay intent” in v1 because salary predictions/associations are not reliable enough to lead recommendations.

---

## 3) Output Buckets (v1)
We output **two** ranked lists:

### Bucket 1 — Best-now
High suitability + lower competitiveness (more reachable).

### Bucket 2 — Stretch
High suitability + higher competitiveness (aligned but harder).

We do NOT include a separate “high-pay” bucket in v1 because salary is already integrated into the competitiveness index in Chapter 3.

---

## 4) Bucket Eligibility Rules (Hard Thresholds)
We use hard thresholds (not percentile splits) to avoid “top X% of poor options” behavior under tight constraints.

### Suitability threshold: adaptive with floor
- Start with `S_min_base = 0.70`
- If candidate count after applying `S_min_base` is below `N_target = 50`, relax suitability threshold in controlled steps down to:
- `S_min_floor = 0.60`
- Never relax below the floor.

### Competitiveness split
- Best-now if: `S >= S_min` AND `C <= C_max`
- Stretch if: `S >= S_min` AND `C > C_max`
- Default: `C_max = 0.50`

---

## 5) Ranking Objective (Within Bucket)
We use transparent lexicographic ranking (no weighted composite score):

1) Higher suitability first  
2) Then lower competitiveness  
3) Deterministic tiebreaks (e.g., stable sort on `job_id` after the above)

---

## 6) Role-Focused Slice Definition (Intent C)
Role-Focused intent uses the minimum viable user input:

- If user provides `job_family_id` (or equivalent), filter to that job family.
- Else, filter by title keyword match on normalized title.
- Users can refine constraints if the role slice is too broad/narrow.

---

## 7) Zero/Low Jobs Policy (No Forced Relaxation Below Floor)
We do not automatically “force” recommendations by relaxing thresholds below the floor.

A policy controls what happens when buckets are empty or too small:
- `zero_jobs_policy = "upskill"` (v1 default)

Meaning: if no jobs (or too few) are returned under constraints and threshold policy, the system pivots to the Upskilling Recommender output as the primary result.

(Other policies like `"requery"` or `"both"` can be added later, but are not the v1 default.)

---

## 8) Explanation Verbosity (User-Configurable)
- `explain_level = "minimal" | "detailed" | "report"`
- Default: `explain_level = "minimal"`

Mapping (v1):
- minimal: suitability, competitiveness, salary percentile (context), top missing skills (top 5)
- detailed: minimal + `why_shown` template string + bucket rationale
- report: detailed + top per-job skill requirement probabilities (top skills only) + gap summary stats

---

## 9) Diversity / De-duplication (v1)
No diversity enforcement in v1.  
Diversity is treated as an emergent property of user constraints:
- if the user wants more diversity, they relax constraints;
- if the domain/location is niche, low diversity is expected and accepted.

---

## 10) Acceptance Checks (v1 Invariants)
1) Determinism: same inputs → same ranked outputs  
2) Constraint-respect: no recommendation violates user filters  
3) Bucket integrity: Best-now has `C <= C_max`; Stretch has `C > C_max` (given enough candidates)  
4) Threshold policy: suitability threshold relaxes only down to `S_min_floor`; never below  
5) Zero-jobs policy: if no jobs, return upskilling output by default (`zero_jobs_policy="upskill"`)
