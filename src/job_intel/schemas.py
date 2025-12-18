# src/job_intel/schemas.py

from datetime import datetime, timezone
from typing import Optional, Union, List, Dict, Any

from src.job_intel.features.userprofile_skill_processing import user_skill_processor


# ------------------------------------------------------------------
# Frozen allow-lists (sourced from Chapter 1 training data)
# ------------------------------------------------------------------

ALLOWED_SECTORS = [
    "Travel & Tourism",
    "Consumer Services",
    "Unknown",
    "Information Technology",
    "Business Services",
    "Insurance",
    "Finance",
    "Retail",
    "Media",
    "Restaurants, Bars & Food Services",
    "Agriculture & Forestry",
    "Non-Profit",
    "Education",
    "Government",
    "Health Care",
    "Oil, Gas, Energy & Utilities",
    "Accounting & Legal",
    "Manufacturing",
    "Real Estate",
    "Biotech & Pharmaceuticals",
    "Arts, Entertainment & Recreation",
    "Aerospace & Defense",
    "Construction, Repair & Maintenance",
    "Transportation & Logistics",
    "Telecommunications",
    "Mining & Metals",
]

ALLOWED_STATES = {
    "NY",
    "NJ",
    "CA",
    "IL",
    "TX",
    "AZ",
    "DE",
    "PA",
    "FL",
    "OH",
    "UT",
    "VA",
    "NC",
    "SC",
    "IN",
    "WA",
    "GA",
    "KS",
    "CO",
    "international",
}

ALLOWED_TITLE_RICH = [
    "general_data_data_scientist",
    "business_data_scientist",
    "general_data_data_analyst",
    "research_scientist",
    "ML_AI_scientist",
    "ML_AI_data_scientist",
    "health_data_engineer",
    "business_data_analyst",
    "general_data_data_engineer",
    "health_data_scientist",
    "ML_AI_data_engineer",
    "business_data_engineer",
    "sport_data_scientist",
    "research_data_analyst",
    "sport_data_analyst",
    "health_scientist",
    "health_data_analyst",
    "research_data_scientist",
    "security_data_scientist",
    "ML_AI_data_analyst",
    "business_scientist",
    "security_data_analyst",
    "security_scientist",
    "general_data_scientist",
    "security_data_engineer",
]

ALLOWED_TITLE_FAMILY = [
    "data_scientist",
    "data_analyst",
    "scientist",
    "data_engineer",
]


# ------------------------------------------------------------------
# UserProfile constructor
# ------------------------------------------------------------------


def build_user_profile(
    skill_text: str = "",
    current_state: Optional[str] = "ALL",
    job_title_family: Optional[str] = None,
    job_title_rich: Optional[str] = None,
    target_sectors: Optional[Union[str, List[str]]] = None,
    salary_target: Optional[Union[int, float, str]] = None,
    explain_skills: bool = False,
    schema_version: str = "v01",
) -> Dict[str, Any]:

    # --------------------------------------------------------------
    # Normalize state
    # --------------------------------------------------------------
    if (
        current_state is None
        or str(current_state).strip() == ""
        or str(current_state).upper() == "ALL"
    ):
        current_state = None
    else:
        state_norm = current_state.strip()
        if state_norm.lower() == "international":
            current_state = "international"
        else:
            current_state = state_norm.upper()

        if current_state not in ALLOWED_STATES:
            raise ValueError(
                f"Unknown state '{current_state}'. Must be one of: {sorted(ALLOWED_STATES)}"
            )

    # --------------------------------------------------------------
    # Normalize and validate title filters
    # --------------------------------------------------------------
    if job_title_family is not None:
        if job_title_family not in ALLOWED_TITLE_FAMILY:
            raise ValueError(
                f"Unknown job_title_family '{job_title_family}'. "
                f"Must be one of: {ALLOWED_TITLE_FAMILY}"
            )

    if job_title_rich is not None:
        if job_title_rich not in ALLOWED_TITLE_RICH:
            raise ValueError(
                f"Unknown job_title_rich '{job_title_rich}'. "
                f"Must be one of: {ALLOWED_TITLE_RICH}"
            )

    # --------------------------------------------------------------
    # Normalize and validate sectors
    # --------------------------------------------------------------
    if isinstance(target_sectors, str):
        target_sectors = [target_sectors]

    if target_sectors is not None:
        if not isinstance(target_sectors, list):
            raise TypeError(
                "target_sectors must be None, a string, or a list of strings"
            )

        target_sectors = [s.strip() for s in target_sectors]

        invalid = [s for s in target_sectors if s not in ALLOWED_SECTORS]
        if invalid:
            raise ValueError(
                f"Unknown sector(s): {invalid}. "
                f"Must be chosen from: {ALLOWED_SECTORS}"
            )

    # --------------------------------------------------------------
    # Salary target validation (stored raw, not used yet)
    # --------------------------------------------------------------
    if salary_target is not None:
        try:
            salary_target = float(salary_target)
        except (TypeError, ValueError) as e:
            raise TypeError("salary_target must be numeric or None") from e

        if salary_target <= 0:
            raise ValueError("salary_target must be > 0")

    # --------------------------------------------------------------
    # Skill processing (validated primitive)
    # --------------------------------------------------------------
    skills = user_skill_processor(skill_text, explain_skills=explain_skills)

    # --------------------------------------------------------------
    # Assemble UserProfile
    # --------------------------------------------------------------
    profile = {
        "raw_inputs": {
            "skill_text": skill_text,
            "current_state": current_state,
            "job_title_family": job_title_family,
            "job_title_rich": job_title_rich,
            "target_sectors": target_sectors,
            "salary_target": salary_target,
        },
        "derived": {
            "skills_by_group": skills["skill_flags_long"],
            "skill_vector": skills["skill_vector"],
            "skill_pcs": skills["skill_pcs"],
            "skill_explanations": skills.get("skill_explanations"),
        },
        "meta": {
            "schema_version": schema_version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    return profile
