# src/job_intel/features/titles.py

from __future__ import annotations

from typing import Optional, Tuple

import re
import pandas as pd

# Retrieve the function from the seniority.py file
from src.job_intel.features.seniority import (
    detect_seniority_from_title,
    detect_seniority_from_description,
    combine_seniority,
)


# ---------------------------------------------------------------------
# 1. CONSTANTS / WORD LISTS
# ---------------------------------------------------------------------

noise_terms = [
    # -------------------------------------------------
    # Contract / job-advert / recruiting meta
    # -------------------------------------------------
    "contract",
    "temporary",
    "temp",
    "full time",
    "part time",
    "fulltime",
    "full",
    "freelance",
    "seasonal",
    "hire",
    "hiring",
    "opportunity",
    "opportunities",
    "urgent",
    "immediate",
    "immediate start",
    "asap",
    "apply now",
    "now hiring",
    "opening",
    "openings",
    "job opening",
    "position",
    "positions",
    "post",
    "posting",
    "reposted",
    "itjobs",
    "jobs",
    "job",
    "careers",
    "candidate",
    # -------------------------------------------------
    # Work mode / location-independent pattern
    # -------------------------------------------------
    "remote",
    "hybrid",
    "onsite",
    "on-site",
    "wfh",
    "work from home",
    "telecommute",
    # -------------------------------------------------
    # Time / schedule / seniority-like ordinals
    # (we handle roman numerals separately)
    # -------------------------------------------------
    "month",
    "months",
    "year",
    "years",
    "day",
    "days",
    "week",
    "weeks",
    "hours",
    "shift",
    "roster",
    "tuesday",
    "saturday",
    "weekend",
    "weekends",
    "weekly",
    "evenings",
    "evening",
    "nights",
    "night",
    "2nd",
    "3rd",
    "1st",
    "pm",
    "am",
    "00pm",
    "30pm",
    "yr",
    "yrs",
    "late",
    "stage",
    # -------------------------------------------------
    # Eligibility / HR boilerplate / generic HR meta
    # -------------------------------------------------
    "relocation",
    "eligibility",
    "visa",
    "sponsorship",
    "anywhere",
    "compensation",
    "resident",
    "candidates",
    "required",
    "must",
    "incentives",
    "proficiency",
    "identity",
    "enrollment",
    "permanent",
    # -------------------------------------------------
    # Generic stopwords (safe to strip from titles)
    # -------------------------------------------------
    "and",
    "or",
    "with",
    "in",
    "of",
    "for",
    "to",
    "the",
    "at",
    "only",
    "on",
    "is",
    "up",
    "may",
    "as",
    "all",
    "two",
    "my",
    "re",
    "from",
    "into",
    "over",
    # -------------------------------------------------
    # Junk / corrupted tokens
    # -------------------------------------------------
    "s",
    "w",
    "t",
    "f",
    "bm",
    "ts",
    "â",
    "area4",
    "_",
    "_long",
    "d",
    "‰øùÂÆû‰π†‰øùÂ∞±‰∏ö",
    "x",
    "ch",
    "ihha",
    "ctj",
    "hsv",
    "mbse",
    "dw",
    "edw",
    "lrec",
    "idq",
    "c3",
    "e3",
    "1q",
    # -------------------------------------------------
    # Pure location tokens (we have a location column)
    # -------------------------------------------------
    # US states
    "ny",
    "nj",
    "ct",
    "fl",
    "oh",
    "pa",
    "tx",
    "mn",
    "wi",
    "nc",
    "sc",
    "al",
    "tn",
    "ky",
    "va",
    "wv",
    "md",
    "de",
    # Cities, regions, countries, generic geo
    "dallas",
    "chicago",
    "austin",
    "atlanta",
    "toronto",
    "washington",
    "antonio",
    "moines",
    "louis",
    "columbus",
    "nationwide",
    "ca",
    "san",
    "us",
    "la",
    "phoenix",
    "diego",
    "houston",
    "irving",
    "philadelphia",
    "francisco",
    "jolla",
    "south",
    "jacksonville",
    "atx",
    "local",
    "jersey",
    "jose",
    "ga",
    "california",
    "los",
    "angeles",
    "usa",
    "square",
    "sfo",
    "oklahoma",
    "bay",
    "nyc",
    "sf",
    "charlotte",
    "america",
    "york",
    "dc",
    "chandler",
    "canada",
    "tempe",
    "fargo",
    "wells",
    "east",
    "lake",
    "sinai",
    "harborside",
    "international",
    "mount",
    "missouri",
    "stafford",
    "sugar",
    "land",
    "arizona",
    "americas",
    "glendale",
    "scottsdale",
    "mesa",
    "gilbert",
    "burbank",
    "culver",
    "europe",
    "aurora",
    "arlington",
    "plano",
    "grand",
    "prairie",
    "wilmington",
    "lawson",
    "uk",
    "greenwood",
    "ohio",
    "india",
    "texas",
    # Extra locations / geo-ish
    "seattle",
    "denver",
    "norfolk",
    "whittier",
    "woodridge",
    "cerritos",
    "glenview",
    "bellaire",
    "hills",
    "hickory",
    "beach",
    "village",
    "acorn",
    "centennial",
    "midwest",
    "west",
    "ocean",
    # -------------------------------------------------
    # Languages / language requirements (not role type)
    # -------------------------------------------------
    "english",
    "korean",
    "german",
    "language",
    "chinese",
    "spanish",
    "bilingual",
    "japanese",
    "indonesian",
    "fluent",
    "speaking",
    # -------------------------------------------------
    # Companies / organisations / brand names
    # (not role semantics)
    # -------------------------------------------------
    "disney",
    "hbo",
    "star",
    "max",
    "apple",
    "ibm",
    "salesforce",
    "oracle",
    "fleishmanhillard",
    "veritystream",
    "epic",
    "tibco",
    "abbott",
    "dell",
    "gao",
    "cowen",
    "pfk",
    "usdc",
    "provost",
    "quantumblack",
    "fbi",
    "google",
    "outsights",
    "lacera",
    "kaiser",
    "tyson",
    "bernard",
    "ais",
    "avmed",
    "hutchinson",
    "teleflora",
    "facilitysource",
    "publicis",
    "wirecutter",
    "peacock",
    "anderson",
    "kapil",
    "bhalla",
    "cedars",
    "svendsen",
    "novi",
    "gtech",
    "point72",
    "acurian",
    "ppd",
    "horsham",
    "schaeffer",
    "atlantic",
    "microsoft",
    "hctra",
    "gnhcc",
    "qsight",
    "sdsa",
    "docutech",
    "qliksense",
    "woocommerce",
    "applecare",
    "amazon",
    "trello",
    "sdma",
    "upstart",
    "tealium",
    "lilly",
    "cognizant",
    "jll",
    "deloitte",
    "cooper",
    "spotify",
    "mongo",
    # -------------------------------------------------
    # SKILLS / STACK TERMS REMOVED FROM TITLE
    # (we keep this section so we can reuse the list
    #  for skill extraction later, but they are stripped
    #  from the job title text used for embeddings)
    # -------------------------------------------------
    # Cloud / data platforms
    "azure",
    "sap",
    "teradata",
    "workday",
    "salesforce",
    "splunk",
    "hana",
    "collibra",
    "mongodb",
    "oracle",
    "hive",
    "snowflake",
    "databricks",
    "powerbi",
    "kafka",
    "kubernetes",
    "docker",
    "talend",
    "geospatial",
    "arcgis",
    "erp",
    "lims",
    "cloudera",
    "greenplum",
    "memsql",
    "celonis",
    "websphere",
    "sharepoint",
    "infosphere",
    "hpc",
    # Programming languages / frameworks
    "python",
    "java",
    "r",
    "sas",
    "nlp",
    "plsql",
    "sql",
    "xml",
    "csv",
    "js",
    # Data / ML / devops
    "etl",
    "cloud",
    "hadoop",
    "spark",
    "crm",
    "mdm",
    "aws",
    "devops",
    "cloudops",
    "pyspark",
    "datamodeling",
    # Misc tools / tech-y but not core role type
    "guidewire",
    "hub",
    "ecommerce",
    "datastage",
    "stack",
    "informatica",
    "pcr",
    "hplc",
    "gpu",
    "polygraph",
    "nifi",
    "elasticsearch",
    "logstash",
    "portal",
    "sdk",
    "cli",
    "scala",
    "ecomm",
    "systemc",
    "oci",
    "ci",
    "cd",
    "datalakes",
    "kibana",
    "tableau",
    "toad",
    "qliksense",
    "spotfire",
    "jupyter",
    "notebook",
    "server",
    "app",
    "apps",
    "node",
    "cpu",
    "graphics",
    # -------------------------------------------------
    # Tasks / processes / generic actions
    # -------------------------------------------------
    "extract",
    "import",
    "set",
    "start",
    # -------------------------------------------------
    # Generic descriptors / fluff not needed in title
    # -------------------------------------------------
    "experienced",
    "online",
    "background",
    "future",
    "professional",
    "summer",
    "work",
    "fall",
    "main",
    "acceptance",
    "affordable",
    "ignite",
    "career",
    "early",
    "assurance",
    "experience",
    "top",
    "new",
    "world",
    "multiple",
    "strong",
    "smart",
    "high",
    "fully",
    "first",
    "fast",
    "moving",
    "extensive",
    # -------------------------------------------------
    # ID codes / acronyms / job ids
    # -------------------------------------------------
    "w2",
    "c2c",
    "c4i",
    "70445br",
    "66201br",
    "jr1013810",
    "d3529",
    "ek020719a",
    "addm04",
    "s02268p",
    "7199u",
    "kp4457495",
    "84r",
    "daqa07",
    "jc2001",
    "jc2002",
    "bhjob11946_",
    "txho_",
    "opening_type",
    "null",
    "1yr",
    "150k",
    "1s",
    "pk",
    "ht",
    "mt",
    "sd",
    "mm",
    "pp",
    "qm",
    "rq",
    "dq",
    "mg",
    "cbch",
    "fhc",
    "cij",
    "x12",
    "euv",
    "ec",
    "usmtf",
    "niwc",
    "sisw",
    # -------------------------------------------------
    # Misc advert / presentation / filler words
    # -------------------------------------------------
    "now",
    "exciting",
    "needed",
    "please",
    "home",
    "things",
    "cities",
    "begin",
    "also",
    "works",
    "bus",
    "ab",
    "go",
    "etc",
    "items",
    "item",
    "amazing",
    "superstar",
    "description",
    # -------------------------------------------------
    # Misc acronyms with no title meaning
    # -------------------------------------------------
    "eib",
    "rm",
    "pbm",
    "ams",
    "edss",
    "mpk",
    "cecl",
    "lll",
    "gsm",
    "mgt",
    "sni",
    "fr",
    "cao",
    "fs",
    "ww",
    "bfsi",
    "yarn",
    "em",
    "dads06",
    "ipp",
    "ho",
    "dm",
    "ut",
    "noke",
    "cpt",
    "dss",
    "pci",
    "vbp",
    "dlp",
    "svcs",
    "usgaap",
    "ccpa",
    "sd",
    "mm",
    "pp",
    "rt",
    "jl",
    "ll",
    "cbo",
    "hcc",
    "usc",
    "sro",
    "spg",
    "sbg",
    "dps",
    "bsa",
    "hla",
    "rp",
    "clc",
    "ssa",
    "mtv",
    # -------------------------------------------------
    # Social / consumer platforms (not title semantics)
    # -------------------------------------------------
    "twitter",
    "tiktok",
    "whatsapp",
    "instagram",
    # -------------------------------------------------
    # Education / training fluff (not academic roles)
    # -------------------------------------------------
    "thesis",
    "educational",
    "training",
    # -------------------------------------------------
    # Misc extremely generic / filler / junk
    # -------------------------------------------------
    "e",
    "adjustment",
    "across",
    "driven",
    "sm",
    "online",
    "site",
    "space",
    "stage",
    "point",
    "line",
    "place",
    "area",
    "inside",
    "next",
    "end",
    "back",
    "round",
    "side",
    "look",
    "get",
    "consider",
    "plenty",
    "eventually",
    "over",
    "specifically",
    # -------------------------------------------------
    # Extra generic job-board / advert noise
    # -------------------------------------------------
    "apply",
    "seller",
    "buyer",
    "store",
]


principal = ["principal"]
executive = ["chief"]
staff = ["staff"]
assistant = ["assistant"]
supervisor = ["supervisor", "supervisory"]
lead = ["lead", "team lead", "tech lead", "coordinator"]
senior = ["senior", "sr", "sr."]
manager = [
    "manager",
    "director",
    "head of",
    "head,",
    "head -",
    "vp",
    "vice president",
    "head",
    "vice",
    "officer",
    "avp",
]
mid = ["mid", "mid level", "associate", "midlevel", "intermediate"]
junior = [
    "junior",
    "jr",
    "jr.",
    "entry level",
    "graduate",
    "grad",
    "entry",
    "undergraduate",
    "internship",
    "intern",
    "trainee",
    "unpaid",
]
seniority_tokens = [
    # principal / lead tiers
    "principal",
    "lead",
    "team lead",
    "tech lead",
    # senior
    "senior",
    "sr",
    "sr.",
    # manager / director / vp
    "manager",
    "director",
    "head of",
    "head,",
    "head -",
    "vp",
    "vice president",
    "head",
    "vice",
    "officer",
    "avp",
    # mid
    "mid",
    "mid level",
    "associate",
    "midlevel",
    "intermediate",
    # junior
    "junior",
    "jr",
    "jr.",
    "entry",
    "entry level",
    "graduate",
    "grad",
    "undergraduate",
    "trainee",
    "unpaid",
    # executive
    "chief",
    "staff",
    "supervisor",
    "supervisory",
    "assistant",
    "coordinator",
    "internship",
    "intern",
    " i ",
    "ii",
    "iii",
    "iv",
    "v",
]

typo_map = {
    "sceintist": "scientist",
    "scientist_": "scientist",
    "ebgineer": "engineer",
    "behavorial": "behavioral",
    "cinical": "clinical",
    "speciliest": "specialist",
    "reseach": "research",
    "automaion": "automation",
    "industial": "industrial",
    "billingual": "bilingual",
    "pharmacovgilance": "pharmacovigilance",
    "leadeship": "leadership",
}


normalisation_rules = [
    # Analyst family: analyst / analysts / analysis / analytic / analytics / analytical
    (r"\b(analyst|analysts|analysis|analytic|analytics|analytical)\b", "analyst"),
    # Scientist family: scientist / scientists / sciences / scientistfoundational / science
    (
        r"\b(scientistfoundational|scientists?|sciences?|science|researcher|research|modeler|modeller|modeling|modelling)\b",
        "scientist",
    ),
    # Engineer / engineering
    (
        r"\b(engineer|engineers|engineering|architect|architecture|developer|programmer[s]?)\b",
        "engineer",
    ),
    # AI / ML / deep learning / machine learning
    (r"\b(artificial intelligence|ai)\b", "ai"),
    (r"\b(ml|machine learning|deep learning|learning|machine)\b", "ml"),
]

# Build regex patterns from the lists
noise_pattern = r"\b(" + "|".join(re.escape(t) for t in noise_terms) + r")\b"
seniority_pattern = r"\b(" + "|".join(re.escape(t) for t in seniority_tokens) + r")\b"


# ---------------------------------------------------------------------
# 2. CORE FUNCTIONS (copied / slightly adapted from notebook code)
# ---------------------------------------------------------------------


def clean_job_title(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Clean a raw job title:
    - lowercases
    - removes noise / punctuation / numbers
    - strips seniority words/phrases
    - returns (base_title, seniority_from_roman_numerals)

    Returns:
        (cleaned_title: str | None, roman_seniority: str | None)
    """
    if pd.isna(text):
        return text, None

    # 0) normalise case + pad with spaces
    s = f" {str(text).lower().strip()} "

    # 1) normalise common separators early so 'i/ii' becomes 'i ii'
    s = re.sub(r"[-/|\\,]", " ", s)

    # 2) detect Roman numeral seniority (longest to shortest)
    seniority: Optional[str] = None
    if re.search(r"\bv\b", s):
        seniority = "principal"
    elif re.search(r"\biv\b", s):
        seniority = "lead"
    elif re.search(r"\biii\b", s):
        seniority = "senior"
    elif re.search(r"\bii\b", s):
        seniority = "mid"
    elif re.search(r"\bi\b", s):
        seniority = "junior"

    # 3) remove Roman numeral tokens from the text regardless
    s = re.sub(r"\b(v|iv|iii|ii|i)\b", " ", s)

    # 4) remove noise words/phrases
    s = re.sub(noise_pattern, " ", s)

    # 5) separators, brackets (run again is harmless)
    s = re.sub(r"[-/|\\,]", " ", s)
    s = re.sub(r"[()\[\]{}]", " ", s)

    # 6) remove seniority words/phrases (non-Roman, e.g. 'senior', 'lead')
    s = re.sub(seniority_pattern, " ", s)

    # 7) punctuation → space
    s = re.sub(r"[^\w\s]", " ", s)

    # 8) isolated numbers
    s = re.sub(r"\b\d+\b", " ", s)

    # 9) remove specific junk phrase(s) if you had any
    s = re.sub(r"\bover experience\b", " ", s)

    # 10) typo corrections (using your typo_map)
    for wrong, correct in typo_map.items():
        s = re.sub(rf"\b{re.escape(wrong)}\b", correct, s)

    # 11) collapse whitespace and strip padding
    s = " ".join(s.split())

    return s, seniority


# normalise_job_family and collapse_family are copied almost verbatim


def normalise_job_family(text: str) -> str:
    """
    Reduce a cleaned base title to a normalised 'family token string'
    like 'data scientist', 'data engineer', 'data analyst', etc.
    """
    if not isinstance(text, str):
        return text

    s = text.lower()

    # your `normalisation_rules` list of (pattern, replacement) from the notebook
    for pattern, repl in normalisation_rules:
        s = re.sub(pattern, repl, s)

    vocab = {"analyst", "scientist", "engineer", "ai", "ml", "data"}
    tokens = [t for t in s.split() if t in vocab]

    seen = []
    for t in tokens:
        if t not in seen:
            seen.append(t)

    return " ".join(seen)


def collapse_family(family_str: str) -> str:
    """
    Collapse 'data scientist engineer analyst' -> 'data_scientist', etc.
    Logic copied from the notebook.
    """
    if not isinstance(family_str, str):
        return ""

    tokens = set(family_str.split())

    if "data" in tokens and "scientist" in tokens:
        return "data_scientist"
    elif "scientist" in tokens:
        return "scientist"
    elif "data" in tokens and "engineer" in tokens:
        return "data_engineer"
    elif "engineer" in tokens:
        return "data_engineer"
    elif "data" in tokens and "analyst" in tokens:
        return "data_analyst"
    elif "analyst" in tokens:
        return "data_analyst"
    elif "data" in tokens:
        return "data_analyst"
    elif "ml" in tokens:
        return "machine_learning"
    elif "ai" in tokens:
        return "artificial_intelligence"
    else:
        return "other"


# ---------------------------------------------------------------------
# 3. PUBLIC ENTRYPOINT FOR PIPELINES
# ---------------------------------------------------------------------


def add_title_features(
    df: pd.DataFrame,
    *,
    title_col: str = "Job Title",
    description_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add all Chapter 0 title-related features to the dataframe.

    Creates:
        - job_title_raw
        - job_title_base
        - seniority_roman
        - seniority_title
        - job_title_norm
        - job_title_family
        - (optionally) seniority_description
        - seniority_combined
    """
    df = df.copy()

    # raw title
    df["job_title_raw"] = df[title_col]

    # base title + seniority from Roman numerals
    clean_pairs = df["job_title_raw"].apply(clean_job_title)
    df["job_title_base"] = clean_pairs.str[0]
    df["seniority_roman"] = clean_pairs.str[1]

    # combine Roman seniority with keyword-based seniority from the title
    df["seniority_title"] = df.apply(
        lambda row: (
            row["seniority_roman"]
            if row["seniority_roman"] is not None
            else detect_seniority_from_title(row["job_title_raw"])
        ),
        axis=1,
    )

    # optional description-based seniority (Notebook 02 logic)
    if description_col is not None:
        df["seniority_description"] = (
            df[description_col].fillna("").apply(detect_seniority_from_description)
        )
        df["seniority_combined"] = df.apply(
            lambda row: combine_seniority(
                row["seniority_description"], row["seniority_title"]
            ),
            axis=1,
        )
    else:
        df["seniority_description"] = None
        df["seniority_combined"] = df["seniority_title"]

    # job family normalisation and collapse
    df["job_title_norm"] = df["job_title_base"].apply(normalise_job_family)

    # Align with Notebook 01 SBERT step:
    # any empty / 'nan' job_title_norm -> treat as 'scientist'
    df.loc[df["job_title_norm"].isin(["", "nan"]), "job_title_norm"] = "scientist"

    df["job_title_family"] = df["job_title_norm"].apply(collapse_family)

    return df
