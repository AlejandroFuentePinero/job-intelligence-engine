# NOTEBOOK 00

import sys

print(sys.executable)

from pathlib import Path

project_root = Path().resolve().parent
sys.path.append(str(project_root))
project_root


from src.job_intel.config import RAW_DS_JOBS_FILE, RAW_DA_JOBS_FILE


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


ds = pd.read_csv(RAW_DS_JOBS_FILE)
da = pd.read_csv(RAW_DA_JOBS_FILE)


ds = ds.drop(["Unnamed: 0", "index"], axis=1)
da = da.drop(["Unnamed: 0"], axis=1)

set(ds.columns) == set(da.columns)  # True

df = pd.concat([ds, da], axis=0, ignore_index=True)


# Nan

df = df.replace(-1, np.nan)
df = df.replace("-1", np.nan)

df.isna().apply(lambda x: (sum(x) / (len(df))) * 100).sort_values().plot(kind="bar")
plt.ylabel("% of total data missing")
plt.xlabel("Feature")
plt.tight_layout()

df.isna().apply(lambda x: (sum(x) / (len(df))) * 100).sort_values(ascending=False)


# Drop Easy Apply and Competitors due to the very large proportion of missing data
df = df.drop(["Easy Apply", "Competitors", "Company Name", "Revenue"], axis=1)


df["state"] = df["Location"].str.extract(r"(?<=,\s)([^,]+)$")


df["state_hq"] = df["Headquarters"].str.extract(r"(?<=,\s)([^,]+)$")
df["state_hq"] = df["state_hq"].str.replace("061", "NY")

df = df.drop(["Location", "Headquarters"], axis=1)


# Pull international listing together
df["state"] = df["state"].apply(
    lambda x: "international" if isinstance(x, str) and len(x) > 2 else x
)
df["state_hq"] = df["state_hq"].apply(
    lambda x: "international" if isinstance(x, str) and len(x) > 2 else x
)

df["sal_clean"] = (
    df["Salary Estimate"]
    .str.replace(" (Glassdoor est.)", "")
    .str.replace("(Glassdoor est.)", "")
    .str.replace("(Employer est.)", "")
    .str.replace("$", "")
    .str.replace("K", "")
    .str.replace(" ", "")
)

df["is_hourly"] = df["sal_clean"].str.contains("PerHour", case=False)
df["sal_clean"] = df["sal_clean"].str.replace("PerHour", "")


df[["sal_min_raw", "sal_max_raw"]] = df["sal_clean"].str.split("-", expand=True)
df["sal_min_raw"] = pd.to_numeric(df["sal_min_raw"], errors="coerce")
df["sal_max_raw"] = pd.to_numeric(df["sal_max_raw"], errors="coerce")

df["sal_min"] = np.where(
    df["is_hourly"], df["sal_min_raw"] * 2080, df["sal_min_raw"] * 1000
)

df["sal_max"] = np.where(
    df["is_hourly"], df["sal_max_raw"] * 2080, df["sal_max_raw"] * 1000
)

df["sal_mean"] = (df["sal_min"] + df["sal_max"]) / 2


df = df.drop(
    ["Salary Estimate", "sal_clean", "is_hourly", "sal_min_raw", "sal_max_raw"], axis=1
)


df["ownership_clean"] = df["Type of ownership"].str.lower()

df["ownership_clean"] = (
    df["ownership_clean"]
    .str.replace("company - private", "private")
    .str.replace("private practice / firm", "private")
    .str.replace("self-employed", "private")
    .str.replace("company - public", "public")
    .str.replace("subsidiary or business segment", "public")
    .str.replace("franchise", "public")
    .str.replace("nonprofit organization", "nonprofit")
    .str.replace("college / university", "nonprofit")
    .str.replace("school / school district", "nonprofit")
    .str.replace("hospital", "nonprofit")
    .str.replace("government", "government")
    # everything else becomes unknown
    .apply(
        lambda x: (
            x
            if x in ["private", "public", "nonprofit", "government", "unknown"]
            else "unknown"
        )
    )
)
df = df.drop("Type of ownership", axis=1)


df["Founded"] = pd.to_numeric(df["Founded"], errors="coerce").astype("Int64")

df.isna().apply(lambda x: (sum(x) / (len(df))) * 100).sort_values().plot(kind="bar")
plt.ylabel("% of total data missing")
plt.xlabel("Feature")
plt.tight_layout()

df.isna().apply(lambda x: (sum(x) / (len(df))) * 100).sort_values(ascending=False)


sns.countplot(
    data=df,
    x="state",
    color="grey",
    edgecolor="black",
    order=df["state"].value_counts().index,
)
plt.xticks(rotation=90)
plt.xlabel("State / Country")
plt.ylabel("Count")
plt.title("Density of job locations")
plt.tight_layout()


plt.figure(figsize=(12, 6))
sns.countplot(
    data=df,
    x="state_hq",
    color="grey",
    edgecolor="black",
    order=df["state_hq"].value_counts().index,
)
plt.xticks(rotation=90)
plt.xlabel("State / Country")
plt.ylabel("Count")
plt.title("Density of Headquarters locations")
plt.tight_layout()


sns.histplot(data=df, x="sal_min", color="grey", edgecolor="black", bins=50, kde=True)

plt.xlabel("Minium salary")
plt.ylabel("Count")
plt.title("Minimum Salary Distribution")
plt.tight_layout()


sns.histplot(
    data=df,
    x="sal_max",
    color="grey",
    edgecolor="black",
    bins=50,
    kde=True,
)

plt.xlabel("Maximum salary")
plt.ylabel("Count")
plt.title("Maximum Salary Distribution")
plt.tight_layout()

sns.histplot(
    data=df,
    x="sal_mean",
    color="grey",
    edgecolor="black",
    bins=50,
    kde=True,
)

plt.xlabel("Mean salary")
plt.ylabel("Count")
plt.title("Average Salary Distribution")
plt.tight_layout()


sns.countplot(
    data=df,
    x="ownership_clean",
    color="grey",
    edgecolor="black",
    order=df["ownership_clean"].value_counts().index,
)
plt.xticks(rotation=45)
plt.xlabel("Ownership")
plt.ylabel("Cout")
plt.title("Ownership spread")
plt.tight_layout()


sns.countplot(
    data=df,
    x="Size",
    color="grey",
    edgecolor="black",
    order=df["Size"].value_counts().index,
)
plt.xticks(rotation=90)
plt.xlabel("Company size")
plt.ylabel("Count")
plt.title("Company size categories")
plt.tight_layout()


sns.histplot(data=df, x="Rating", color="grey", edgecolor="black", bins=30, kde=True)
plt.xticks(rotation=90)
plt.xlabel("Job rating")
plt.ylabel("Count")
plt.title("Rating distribution")
plt.tight_layout()


sns.countplot(
    data=df,
    x="Sector",
    color="grey",
    edgecolor="black",
    order=df["Sector"].value_counts().index,
)
plt.xticks(rotation=90)
plt.xlabel("")
plt.ylabel("Count")
plt.title("Sector segregation")
plt.tight_layout()


sns.histplot(data=df, x="Founded", color="grey", edgecolor="black", bins=30, kde=True)
plt.xlabel("Year founded")
plt.ylabel("Count")
plt.title("Company distribution by year founded")
plt.tight_layout()


df.plot(kind="scatter", x="Rating", y="sal_mean", alpha=0.2)


from src.job_intel.config import INTERIM_DATA_DIR

INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(INTERIM_DATA_DIR / "01_cleaned_jobs_interim.csv", index=False)


# NOTEBOOK 01

import sys

print(sys.executable)


from pathlib import Path

project_root = Path().resolve().parent
sys.path.append(str(project_root))
project_root


from src.job_intel.config import INTERIM_CLEANED_JOBS

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter


df = pd.read_csv(INTERIM_CLEANED_JOBS)


df["job_title_raw"] = df["Job Title"].astype(str).str.strip()
print("Number of rows:", len(df))
print("Unique raw titles:", df["job_title_raw"].nunique())
print("\nSample titles:")
print(df["job_title_raw"].sample(20, random_state=42).to_list())


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


# Seniority tokens (conservative, lowercase)
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


def detect_seniority(text):
    x = text.lower()

    if any(k in x for k in principal):
        return "principal"
    if any(k in x for k in lead):
        return "lead"
    if any(k in x for k in senior):
        return "senior"
    if any(k in x for k in manager):
        return "manager"
    if any(k in x for k in mid):
        return "mid"
    if any(k in x for k in junior):
        return "junior"
    if any(k in x for k in executive):
        return "executive"
    if any(k in x for k in assistant):
        return "assistant"
    if any(k in x for k in staff):
        return "staff"
    if any(k in x for k in supervisor):
        return "supervisor"

    return "unknown"


# Build regex patterns from my existing lists/dicts
noise_pattern = r"\b(" + "|".join(re.escape(t) for t in noise_terms) + r")\b"
seniority_pattern = r"\b(" + "|".join(re.escape(t) for t in seniority_tokens) + r")\b"


def clean_job_title(text):

    if pd.isna(text):
        return text, None

    # 0) normalise case + pad with spaces
    text = f" {str(text).lower().strip()} "

    # 1) normalise common separators early so 'i/ii' becomes 'i ii'
    text = re.sub(r"[-/|\\,]", " ", text)

    # 2) detect Roman numeral seniority (longest to shortest)
    seniority = None
    if re.search(r"\bv\b", text):
        seniority = "principal"
    elif re.search(r"\biv\b", text):
        seniority = "lead"
    elif re.search(r"\biii\b", text):
        seniority = "senior"
    elif re.search(r"\bii\b", text):
        seniority = "mid"
    elif re.search(r"\bi\b", text):
        seniority = "junior"

    # 3) remove Roman numeral tokens from the text regardless
    text = re.sub(r"\b(v|iv|iii|ii|i)\b", " ", text)

    # 4) remove noise words/phrases
    text = re.sub(noise_pattern, " ", text)

    # 5) separators, brackets (run again is harmless)
    text = re.sub(r"[-/|\\,]", " ", text)
    text = re.sub(r"[()\[\]{}]", " ", text)

    # 6) remove seniority words/phrases (non-Roman, e.g. 'senior', 'lead')
    text = re.sub(seniority_pattern, " ", text)

    # 7) punctuation → space
    text = re.sub(r"[^\w\s]", " ", text)

    # 8) isolated numbers
    text = re.sub(r"\b\d+\b", " ", text)

    # 9) remove specific junk phrase
    text = re.sub(r"\bover experience\b", " ", text)

    # 10) typo corrections
    for wrong, correct in typo_map.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", correct, text)

    # 11) collapse whitespace and strip padding
    text = " ".join(text.split())

    return text, seniority


df["clean_title_and_roman"] = df["job_title_raw"].apply(clean_job_title)
df["job_title_base"] = df["clean_title_and_roman"].apply(lambda x: x[0])
df["seniority_roman"] = df["clean_title_and_roman"].apply(lambda x: x[1])

# combine roman seniority with keyword-based seniority
df["seniority"] = df.apply(
    lambda row: (
        row["seniority_roman"]
        if row["seniority_roman"] is not None
        else detect_seniority(row["job_title_raw"])
    ),
    axis=1,
)


order = df["seniority"].value_counts().index

sns.countplot(data=df, x="seniority", order=order)
plt.xticks(rotation=90)
plt.tight_layout()


# Combine all titles into one long list of words
all_words = " ".join(df["job_title_base"]).split()

# Count frequencies
word_counts = Counter(all_words)

# Convert to a DataFrame for easy viewing
bow = pd.DataFrame(word_counts.items(), columns=["word", "count"]).sort_values(
    "count", ascending=False
)

from src.job_intel.config import INTERIM_DATA_DIR

# bow.to_csv(INTERIM_DATA_DIR / "bow_v3.csv", index=False)

len(bow)


df.to_csv(INTERIM_DATA_DIR / "02_job_titles_cleaned.csv", index=False)


unique_titles = df["job_title_base"].astype(str).str.strip().unique().tolist()
len(unique_titles)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

title_embeddings = model.encode(
    unique_titles,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

title_embeddings.shape

# Nearest-neighbour similarity checks
from sentence_transformers.util import cos_sim


def nearest_titles(query, k=10):
    idx = unique_titles.index(query)
    query_emb = title_embeddings[idx]

    sims = cos_sim(query_emb, title_embeddings)[0].cpu().numpy()
    top_idx = sims.argsort()[::-1][1 : k + 1]  # skip itself at index 0

    return [(unique_titles[i], sims[i]) for i in top_idx]


nearest_titles("data engineer", k=10)


# Embedding norms & distribution
np.mean(np.linalg.norm(title_embeddings, axis=1)), np.std(
    np.linalg.norm(title_embeddings, axis=1)
)


# Quick 2D viz
import umap

reducer = umap.UMAP(random_state=42)
emb_2d = reducer.fit_transform(title_embeddings)

plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=5)
plt.title("SBERT Embeddings (2D UMAP)")
plt.tight_layout()


from sklearn.cluster import KMeans

km_model = KMeans(n_clusters=40, random_state=42, n_init="auto")

labels_km = km_model.fit_predict(title_embeddings)
np.unique(labels_km, return_counts=True)


km_df = pd.DataFrame(
    {"job_title_base": unique_titles, "cluster": labels_km}
).sort_values("cluster")


for i in range(0, (km_df["cluster"].max() + 1), 1):
    subset = km_df[km_df["cluster"] == i]
    print(f"Cluster ID: {i}")
    print(subset["job_title_base"].head(30))

    cluster_domains = {
        0: "health",
        1: "business",
        2: "general_data",
        3: "business",
        4: "health",
        5: "business",
        6: "research",
        7: "business",
        8: "ML_AI",
        9: "ML_AI",
        10: "health",
        11: "general_data",
        12: "business",
        13: "business",
        14: "general_data",
        15: "general_data",
        16: "general_data",
        17: "health",
        18: "research",
        19: "business",
        20: "research",
        21: "health",
        22: "health",
        23: "general_data",
        24: "business",
        25: "general_data",
        26: "research",
        27: "business",
        28: "health",
        29: "business",
        30: "sport",
        31: "research",
        32: "research",
        33: "security",
        34: "health",
        35: "ML_AI",
        36: "research",
        37: "research",
        38: "business",
        39: "health",
    }

km_df["domain"] = km_df["cluster"].map(cluster_domains)
km_df.sample(50)[["job_title_base", "cluster", "domain"]]

order = km_df["domain"].value_counts().index

sns.countplot(data=km_df, x="domain", order=order)
plt.xticks(rotation=90)
plt.tight_layout()

df = df.merge(km_df, on="job_title_base", how="left")


# Spot-check a few titles
df[["Job Title", "job_title_base", "cluster", "domain"]].sample(30, random_state=0)

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


def normalise_job_family(text: str) -> str:
    if not isinstance(text, str):
        return text

    s = text.lower()

    # apply each regex rule
    for pattern, repl in normalisation_rules:
        s = re.sub(pattern, repl, s)

    # keep only tokens that are in your family vocabulary
    vocab = {"analyst", "scientist", "engineer", "ai", "ml", "data"}

    tokens = [t for t in s.split() if t in vocab]

    # dedupe but preserve order
    seen = []
    for t in tokens:
        if t not in seen:
            seen.append(t)

    return " ".join(seen)


df["job_title_norm"] = df["job_title_base"].apply(normalise_job_family)

df[df["job_title_norm"] == ""]["job_title_base"]
df.loc[df["job_title_norm"].isin(["", "nan"]), "job_title_norm"] = "scientist"


def collapse_family(family_str: str) -> str:

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


df["job_title_family"] = df["job_title_norm"].apply(collapse_family)
df["job_title_family"].value_counts()


df["job_title_clean"] = df["domain"].astype(str) + " " + df["job_title_family"]
df["job_title_clean"].value_counts()


df.to_csv(INTERIM_DATA_DIR / "03_job_titles_cleaned_domain_and_family.csv", index=False)

# NOTEBOOK 02


import sys
from pathlib import Path

project_root = Path().resolve().parent
sys.path.append(str(project_root))
project_root

from src.job_intel.config import INTERIM_NORM_TITLE

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter

df = pd.read_csv(INTERIM_NORM_TITLE)
df.info()


from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

whitelist_short = {"c", "r", "js", "go", "c++", "c#", "s3", "ab"}


def surface_cleaning(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Split everything on punctuation except + and #
    text = re.sub(r"[^a-z0-9+#]+", " ", text)

    tokens = text.split()
    cleaned = []
    seen = set()

    for tok in tokens:

        tok = tok.strip(".,:;!?")

        # Skip slash-combined tokens e.g. analyst/sql, developer/analyst
        if "/" in tok:
            for part in tok.split("/"):
                if len(part) >= 2:
                    cleaned.append(part)
            continue

        # stopwords
        if tok in stop_words:
            continue

        # length filter except whitelist
        if len(tok) < 3 and tok not in whitelist_short:
            continue

        # digits
        if tok.isdigit() or len(re.findall(r"\d", tok)) >= 3:
            continue

        # alphanumeric garbage
        if re.match(r"^[a-z]+[0-9]+[a-z]+$", tok):
            continue

        # too many punctuation markers
        if sum(c in "+.#" for c in tok) > 2:
            continue

        if tok not in seen:
            cleaned.append(tok)
            seen.add(tok)

    return " ".join(cleaned)


df["job_description_clean"] = df["Job Description"].apply(surface_cleaning)


principal = [r"\bprincipal\b"]

executive = [
    r"\bchief\b",
    r"\bchief [a-z]+ officer\b",
    r"\bvice president\b",
    r"\bsvp\b",
    r"\bvp\b",
    r"\bexecutive\b",
]

assistant = [
    r"\bassistant\b",
]

supervisor = [
    r"\bsupervisor\b",
    r"\bsupervisory\b",
]

lead = [
    r"\bteam lead\b",
    r"\btech lead\b",
]

senior = [
    r"\bsenior\b",
    r"\bsr\b",
    r"\bsr\.\b",
]

manager = [
    r"\bmanager\b",
    r"\bdirector\b",
    r"\bhead of\b",
    r"\bhead[- ]?[\w ]*department\b",
    r"\bofficer\b",
    r"\bavp\b",
]

mid = [
    r"\bmid[- ]?level\b",
    r"\bassociate\b",
    r"\bintermediate\b",
]

junior = [
    r"\bjunior\b",
    r"\bjr\b",
    r"\bjr\.\b",
    r"\bentry level\b",
    r"\binternship\b",
    r"\bintern\b",
    r"\btrainee\b",
    r"\bunpaid\b",
]

# priority order: higher first
seniority_patterns = [
    ("principal", principal),
    ("executive", executive),
    ("manager", manager),
    ("lead", lead),
    ("senior", senior),
    ("mid", mid),
    ("junior", junior),
    ("assistant", assistant),
    ("supervisor", supervisor),
]


def detect_seniority(text):
    if not isinstance(text, str):
        return "unknown"

    x = text.lower()
    # optional: restrict to first N chars to avoid noisy paragraphs
    snippet = x[:300]

    for label, patterns in seniority_patterns:
        for pat in patterns:
            if re.search(pat, snippet):
                return label

    return "unknown"


df["seniority_description"] = (
    df["job_description_clean"].fillna("").apply(detect_seniority)
)


def combine_seniority(desc_val, title_val):
    if desc_val != "unknown":
        return desc_val
    elif title_val != "unknown":
        return title_val
    else:
        return "unknown"


df["seniority_combined"] = df.apply(
    lambda row: combine_seniority(row["seniority_description"], row["seniority"]),
    axis=1,
)


order = df["seniority_combined"].value_counts().index

sns.countplot(data=df, x="seniority_combined", order=order)
plt.xticks(rotation=90)
plt.tight_layout()


df["title_plus_description"] = (
    df["job_title_base"].astype(str) + " " + df["job_description_clean"].astype(str)
)


from src.job_intel.features.skill_extractor import extract_domain_level_flags

text = "We need someone with strong Python and SQL experience."

extract_domain_level_flags(text)

from src.job_intel.features.skill_extractor import explain_matches

explain_matches("We need someone with strong Python and SQL experience.")


from src.job_intel.features.skill_extractor import extract_domain_level_flags

df["skill_flags"] = df["title_plus_description"].apply(extract_domain_level_flags)


features_df = pd.DataFrame(df["skill_flags"].tolist())

df = pd.concat([df, features_df], axis=1)

from src.job_intel.features.skill_extractor import explain_matches

row0_text = df.loc[0, "title_plus_description"]
matches0 = explain_matches(row0_text)
matches0


skill_cols = [c for c in df.columns if "__" in c]

summary = df[skill_cols].sum().sort_values(ascending=False)
summary


from src.job_intel.config import INTERIM_DATA_DIR

df.to_csv(INTERIM_DATA_DIR / "05_skills_extracted.csv", index=False)
